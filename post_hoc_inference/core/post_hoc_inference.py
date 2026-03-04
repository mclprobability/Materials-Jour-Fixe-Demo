import arviz as az
import pyro
import pyro.infer
import pyro.infer.autoguide
import pyro.optim
import sympy as sp
import torch
from pyro.infer import (
    MCMC,
    NUTS,
    JitTrace_ELBO,
    Predictive,
    RandomWalkKernel,
    Trace_ELBO,
)
from pyro.infer.autoguide import init_to_value
from pyro.infer.mcmc.util import select_samples

from post_hoc_inference.core.pyro_backend import sympy_to_pyro as sp_to_pyro
from post_hoc_inference.core.pyro_backend import utils as pyro_utils
from post_hoc_inference.core.sympy_backend import utils as sp_utils
from post_hoc_inference.utils import log

logger = log.getLogger(__name__)


class PostHocInference:
    """A scikit-learn style estimator for performing Bayesian post-hoc inference on deterministic equations.

    This class bridges the gap between symbolic mathematics and probabilistic machine learning.
    It takes a deterministic mathematical model (defined as a SymPy expression), automatically
    identifies its numeric constants, and replaces them with learnable parameters. It then
    compiles this symbolic expression into a probabilistic Pyro model.

    By providing a familiar API (fit, predict, score), users can easily apply Markov Chain
    Monte Carlo (MCMC) methods or Variational Inference (VI) to fit the symbolic model to empirical data. This allows users
    to not only find optimal parameter values, but also quantify uncertainty, extract posterior
    distributions, and generate predictive distributions with confidence bounds.

    Typical Use Cases:
        - Calibrating physics-based or first-principles equations with noisy data.
        - Adding uncertainty quantification to equations discovered via Symbolic Regression.
        - Evaluating the robustness of a mathematical model's parameters.
    """

    def __init__(
        self,
        expr: sp.Expr,
        input_symbols: list[sp.Symbol],
        output_dim=1,
        inference_params: dict = {},
    ) -> None:
        """Initializes the PostHocInference model for Bayesian inference on SymPy expressions.

        Args:
            expr (sp.Expr): The symbolic mathematical expression representing the core model structure.
            input_symbols (list[sp.Symbol]): A list of SymPy symbols corresponding to the input features (independent variables).
            output_dim (int, optional): The dimensionality of the model's output. Defaults to 1.
            inference_params (dict, optional): Configuration parameters for the MCMC sampling (e.g., 'n_chains': int, 'n_samples': int, 'n_warmup_samples': int, 'kernel_name': ('random_walk', 'nuts')). Defaults to {}.
        """
        self._expr_sp = expr
        self._input_symbols = input_symbols
        self._inference_params = inference_params
        self._mcmc = None
        self._svi = None

        self._expr_sp_parameterized, self._exp_param_values = (
            sp_utils.replace_constants_with_parameters(self._expr_sp)
        )
        self._pyro_model = sp_to_pyro.create_pyro_model(
            self._expr_sp_parameterized,
            self._input_symbols,
            list(self._exp_param_values.keys()),
            output_dim=output_dim,
        )
        self._setup_inference()

    def _setup_inference(self) -> None:
        """Setup Inference - either MCMC or VI.

        Args:
            None
        """
        self._kernel_name = self._inference_params.get("kernel_name", "nuts")
        self._jit_compile = self._inference_params.get("jit_compile", False)
        self._n_chains = self._inference_params.get("n_chains", 1)

        self._initial_params = pyro_utils.get_initial_param_dict(
            self._exp_param_values, n_chains=self._n_chains
        )

        match (self._kernel_name):
            case "nuts" | "random_walk":
                self._setup_mcmc()
            case "vi":
                self._setup_svi()
            case _:
                raise NotImplementedError(
                    f"Kernel '{self._kernel_name}' not implemented. Valid kernels: 'nuts', 'random_walk', 'vi'."
                )

    def _setup_mcmc(self):
        """Configures the MCMC inference kernel and sampler based on the provided initialization parameters."""

        self._n_warmup_samples = self._inference_params.get("n_warmup_samples", 1000)
        self._n_samples = self._inference_params.get("n_samples", 1000)
        initial_step_size = self._inference_params.get("initial_step_size", 1e-2)

        match (self._kernel_name):
            case "nuts":
                self._kernel = NUTS(
                    self._pyro_model,
                    jit_compile=self._jit_compile,
                    step_size=initial_step_size,
                )
            case "random_walk":
                self._kernel = RandomWalkKernel(
                    self._pyro_model, init_step_size=initial_step_size
                )
            case _:
                raise NotImplementedError(
                    f"Kernel '{self._kernel_name}' not implemented. Valid kernels: 'nuts', 'random_walk'."
                )
        self._mcmc = MCMC(
            self._kernel,
            num_samples=self._n_samples,
            warmup_steps=self._n_warmup_samples,
            num_chains=self._n_chains,
            initial_params=self._initial_params,
        )

    def _setup_svi(self):
        """Configures the VI inference kernel and optimizer based on the provided initialization parameters."""

        self._n_vi_iter = self._inference_params.get("vi_iter", 1000)
        self._n_particles = self._inference_params.get("n_particles", 1)

        self._guide = pyro.infer.autoguide.AutoMultivariateNormal(
            self._pyro_model, init_loc_fn=init_to_value(values=self._initial_params)
        )
        self._optim = pyro.optim.ClippedAdam(
            {"lr": 1e-2, "lrd": (1e-2) ** (1 / self._n_vi_iter)}
        )
        if self._jit_compile:
            self._loss = JitTrace_ELBO(
                num_particles=self._n_particles, vectorize_particles=True
            )
        else:
            self._loss = Trace_ELBO(
                num_particles=self._n_particles, vectorize_particles=True
            )
        self._svi = pyro.infer.SVI(
            self._pyro_model, self._guide, self._optim, self._loss
        )

    def fit(self, X: torch.Tensor | None, y: torch.Tensor) -> None | list:
        """Fits the Bayesian model to the provided data using the configured inference method.

        Args:
            X (torch.Tensor | None): The input feature data of shape (batch, features).
            y (torch.Tensor): The target output data of shape (batch, output_dim).
        """
        pyro.clear_param_store()
        if X is not None:
            assert X.dim() == 2 and "X shape must be (batch, features)"
        assert y.dim() == 2 and "y shape must be (batch, features)"
        if self._mcmc is not None:
            self._mcmc.run(X, y)
        elif self._svi is not None:
            self._losses = []
            for i in range(self._n_vi_iter):
                loss = self._svi.step(X, y)
                self._losses.append(loss)
                if i % 100 == 0:
                    logger.debug(f"[SVI] loss {i:4d}: {loss:.4f}")
            return self._losses
        else:
            raise Exception("Either '_mcmc' or '_svi' must be initialized.")

    def predict(
        self,
        X: torch.Tensor | None,
        n_predictive_samples=500,
        sample_prior=False,
        parallel=False,
    ) -> dict[str, torch.Tensor]:
        """Generates predictive samples from the fitted posterior or the unconditioned prior distribution.

        Args:
            X (torch.Tensor | None): The input feature data of shape (batch, features) for which to generate predictions.
            n_predictive_samples (int, optional): The number of predictive samples to draw. Defaults to 500.
            sample_prior (bool, optional): If True, draws samples from the prior instead of the fitted posterior. Defaults to False.
            parallel (bool, optional): If True, vectorize the model and evaluate the posterior samples in parallel. Defaults to False.

        Returns:
            dict: A dictionary containing the predictive samples generated by the underlying Pyro model.
        """
        assert (
            self._mcmc is not None
            or self._svi is not None
            and "Regressor not trained. Run fit() first."
        )
        if X is not None:
            assert X.dim() == 2 and "X shape must be (batch, features)"

        if sample_prior:
            predictive = Predictive(
                self._pyro_model,
                posterior_samples={},
                num_samples=n_predictive_samples,
                parallel=parallel,
            )
        else:
            if self._mcmc is not None:
                predictive = Predictive(
                    self._pyro_model,
                    posterior_samples=select_samples(
                        self._mcmc.get_samples(), n_predictive_samples
                    ),
                    parallel=parallel,
                )
            else:
                predictive = Predictive(
                    self._pyro_model,
                    guide=self._guide,
                    parallel=parallel,
                    num_samples=n_predictive_samples,
                )
        return predictive(X)

    def inference_diagnostics(self, print_summary=True) -> dict:
        """Computes and optionally prints diagnostic statistics for the MCMC sampling run.

        Args:
            print_summary (bool, optional): Whether to print the MCMC summary table to the console. Defaults to True.

        Returns:
            dict: A dictionary containing diagnostic metrics (like effective sample size and Gelman-Rubin statistics).
        """
        if self._mcmc is not None:
            if print_summary:
                self._mcmc.summary()
            return self._mcmc.diagnostics()
        else:
            return {"loss": self._losses}

    def score(self, X, y=None):
        pass

    def get_posterior(self) -> az.InferenceData | torch.distributions.Distribution:
        """If MCMC, extracts the posterior samples as an ArviZ InferenceData object for advanced analysis and plotting.
        If VI, return the variational distribution.

        Returns:
            arviz.InferenceData | torch.distributions.Distribution: The posterior distribution.
        """
        if self._mcmc is not None:
            return az.from_pyro(self._mcmc)
        return self._guide.get_posterior()

    def render_model(self, x_dummy, filename=None):
        """Renders a graphical representation of the Pyro probabilistic model.

        Args:
            x_dummy (torch.Tensor): A dummy input tensor used to trace the model's execution graph.
            filename (str | None, optional): The file path to save the rendered graph (e.g., 'model.png'). Defaults to None.
        """
        pyro.render_model(self._pyro_model, (x_dummy,), filename=filename)
