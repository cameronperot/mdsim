import os
import pickle
from time import time

import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MDSim:
    def __init__(self, config):
        """
        Initialize the system with the variables from the config.
        """
        self.config = config
        self.N = config["N"]
        self.d = config["d"]
        self.m = config["m"]
        self.T = config["T"]
        self.τ = config["τ"]
        self.L = config["L"]
        self.ε = config["ε"]
        self.σ = config["σ"]
        self.k_B = config["k_B"]
        self.seed = config["seed"]
        self.n_iter = config["n_iter"]
        self.ensemble_type = config["ensemble_type"]
        self.rescale_velocity_interval = config["rescale_velocity_interval"]
        self.diag_indices = (np.arange(self.N), np.arange(self.N))
        self.save_path = os.path.abspath(config.get("save_path", "/tmp/mdsim/data"))

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        assert isinstance(self.N, int)
        assert isinstance(self.n_iter, int)
        assert self.d in [2, 3]
        assert self.ensemble_type in ["NVE", "NVT"]

        if "seed" in config:
            np.random.seed(self.seed)

        if "x" in config and "v" in config:
            self.x = config["x"]
            self.v = config["v"]
        else:
            self._initialize_random_positions()
            self._initialize_random_velocities()

        assert self.x.shape == self.v.shape

        self.a = np.zeros(self.x.shape)
        self._compute_accelerations()
        self._compute_E()

    def _initialize_random_positions(self):
        """
        Assign particles evenly spaced positions in a grid [σ, L-σ) with normally
        distributed noise of zero mean and a standard deviation of 0.1
        """
        if self.d == 2:
            x = np.linspace(self.σ / 2, self.L - self.σ / 2, int(np.ceil(np.sqrt(self.N))))
            xx, yy = np.meshgrid(x, x)
            self.x = np.array([xx.flatten(), yy.flatten()]).T
            self.x += np.random.normal(0, 0.01 * self.σ, self.x.shape)
        elif self.d == 3:
            x = np.linspace(
                self.σ / 2, self.L - self.σ / 2, int(np.ceil(self.N ** (1 / 3)))
            )
            xx, yy, zz = np.meshgrid(x, x, x)
            self.x = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T[: self.N]
            self.x += np.random.normal(0, 0.01 * self.σ, self.x.shape)

    def _initialize_random_velocities(self):
        """
        Assign particles random velocities according to the Boltzmann distribution
        """
        μ = 0
        σ = np.sqrt(self.k_B * self.T / self.m)
        self.v = np.random.normal(μ, σ, (self.N, self.d))
        # make sure that there is no center of mass movement
        self.v -= np.mean(self.v, axis=0)
        # rescale the velocity to match the expected kinetic energy at the given T
        self._rescale_velocity()

    def _rescale_velocity(self):
        """
        Rescale the velocity to satisfy the relation ⟨E_kin⟩ = (d/2)(N-1)k_B T
        """
        λ = np.sqrt(
            self.d * self.k_B * (self.N - 1) * self.T / (self.m * np.sum(self.v ** 2))
        )
        self.v *= λ

    def _compute_r_nearest_image(self):
        """
        Computes the set of difference vectors corresponding to the nearest image particles.

        The r_nearest_immage array is structured such that there are N entries of size
        (N-1, d), where the ith element is the array containing the difference vectors
        of the other N - 1 particles' nearest image.

        The r_nearest_image_magnitude array is structured such that
        r_nearest_image_magnitude[i, j] is the magnitude of the difference vector
        r_nearest_image[i, j].

        Note that the indexing does not line exactly up since the array r_nearest_image[i]
        is of length N - 1 due to the lack of an ith element.
        """
        # wrap the particles' positions back into [0, L)^d to compute distances with
        x = self.x % self.L
        # compute x[i] - x[j] for all j, i.e. r_nearest_image[i, j] = x[i] - x[j]
        self.r_nearest_image = x.reshape((self.N, 1, self.d)) - np.repeat(
            x[np.newaxis, :, :], self.N, axis=0
        )
        # if the "left" distance is < L/2, then use the image particle on the "right"
        self.r_nearest_image[self.r_nearest_image > self.L / 2] -= self.L
        # if the "right" distance is > -L/2, then use the image particle on the "left"
        self.r_nearest_image[self.r_nearest_image < -self.L / 2] += self.L

        # compute the magnitudes of the nearest image difference vectors
        self.r_nearest_image_magnitude = np.linalg.norm(
            self.r_nearest_image, axis=-1, keepdims=True
        )
        # set r_nearest_image_magnitude[i, i] = np.inf for all i to avoid div 0 errors in
        # potential/force computations
        self.r_nearest_image_magnitude[self.diag_indices] = np.inf

    def _compute_accelerations(self):
        """
        Compute acceleration based off of current positions and store old acceleration for
        use in computing the velocity.
        """
        self.a_prev = self.a

        self._compute_r_nearest_image()
        α = (self.σ / self.r_nearest_image_magnitude) ** 6
        self.a = (
            np.sum(
                ((48 * self.ε) * (α ** 2 - 0.5 * α) / self.r_nearest_image_magnitude ** 2)
                * self.r_nearest_image,
                axis=1,
            )
            / self.m
        )

    def _compute_E(self):
        """
        Compute the potential and kinetic energies.
        """
        α = (self.σ / self.r_nearest_image_magnitude) ** 6
        self.E_pot = np.sum((4 * self.ε) * (α ** 2 - α)) / 2
        self.E_kin = (self.m / 2) * np.sum(self.v ** 2)
        self.E = self.E_kin + self.E_pot

    def _initialize_history_arrays(self):
        """
        Initialize arrays in which to store the history of the observables.
        """
        self.E_kin_history = np.zeros(self.n_iter + 1)
        self.E_pot_history = np.zeros(self.n_iter + 1)
        self.x_history = np.zeros((self.n_iter + 1, self.N, self.d))
        self.v_history = np.zeros((self.n_iter + 1, self.N, self.d))

        self._update_history_arrays(0)

    def _update_history_arrays(self, iter):
        """
        Set the iter entry of the history arrays with the current observable values.
        """
        self.E_kin_history[iter] = self.E_kin
        self.E_pot_history[iter] = self.E_pot
        self.x_history[iter] = self.x
        self.v_history[iter] = self.v

    def _create_data_dfs(self):
        """
        Create DataFrames containing the data from the simulation.
        """
        self.df_E = pd.DataFrame(
            {
                "t": np.linspace(0, self.n_iter * self.τ, self.n_iter + 1),
                "E_kin": self.E_kin_history,
                "E_pot": self.E_pot_history,
                "E": self.E_kin_history + self.E_pot_history,
            }
        )

        self.df_particles = pd.DataFrame(
            {
                "t": np.repeat(
                    np.linspace(0, self.n_iter * self.τ, self.n_iter + 1), self.N
                ),
                "particle_id": np.resize(
                    np.arange(1, self.N + 1), self.N * (self.n_iter + 1)
                ),
                "x_1": self.x_history[:, :, 0].reshape(-1),
                "x_2": self.x_history[:, :, 1].reshape(-1),
                "v_1": self.v_history[:, :, 0].reshape(-1),
                "v_2": self.v_history[:, :, 1].reshape(-1),
            }
        )
        if self.d == 3:
            self.df_particles["x_3"] = self.x_history[:, :, 2].reshape(-1)
            self.df_particles["v_3"] = self.v_history[:, :, 2].reshape(-1)

        self._save_dfs_to_csv()

    def _save_dfs_to_csv(self):
        """
        Save the DataFrames to CSVs.
        """
        self.df_E.to_csv(os.path.join(self.save_path, "df_E.csv"))
        self.df_particles.to_csv(os.path.join(self.save_path, "df_particles.csv"))

    def _save_config_to_pickle(self):
        """
        Save the system's current state to a pickle.
        """
        state = self.config
        state["x"] = self.x
        state["v"] = self.v

        with open(os.path.join(self.save_path, "state.pkl"), "wb") as f:
            pickle.dump(state, f)

    def plot_system(self, save_as=False, animate=False):
        """
        Plot the particles wrapped back into the box.
        """
        fig, ax = plt.subplots(dpi=144, figsize=(5, 5))
        x = self.x % self.L
        ax.scatter(x[:, 0], x[:, 1], c="b", s=5)
        ax.set_xlim((0, self.L))
        ax.set_ylim((0, self.L))
        ax.set_xlabel(r"$x$ [Å]")
        ax.set_ylabel(r"$y$ [Å]")
        ax.set_aspect("equal")
        plt.tight_layout()

        if save_as:
            plt.savefig(os.path.join(self.save_path, save_as))

        if animate:
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return image

    def velocity_verlet(self, animate=False, save_as="animation.gif", fps=10):
        """
        Run the velocity verlet scheme for n_iter iterations, and update the system's
        variables.
        """
        self._initialize_history_arrays()

        t_0 = time()
        frames = []
        for n in range(self.n_iter):
            self.x += self.τ * self.v + (self.τ ** 2 / 2) * self.a
            self._compute_accelerations()
            self.v += self.τ * (self.a_prev + self.a) / 2

            if n % self.rescale_velocity_interval == 0 and self.ensemble_type == "NVT":
                self._rescale_velocity()
            if n % 1000 == 0:
                print(f"Iteration: {n} / {self.n_iter}")

            self._compute_E()
            self._update_history_arrays(n + 1)

            if animate:
                frames.append(self.plot_system(animate=True))

        print(f"Iteration: {self.n_iter} / {self.n_iter}")
        print(f"Completed {self.n_iter} iterations in {time() - t_0:.3f} seconds.")

        self._create_data_dfs()
        self._save_config_to_pickle()
        if animate:
            imageio.mimsave(os.path.join(self.save_path, save_as), frames, fps=fps)
