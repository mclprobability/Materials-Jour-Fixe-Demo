# Materials-Jour-Fixe-Demo

The **Post-hoc Inference** library converts a deterministic SymPy expression into a probabilistic model and performs Bayesian (MCMC) inference on the expression's numeric constants. This enables uncertainty quantification over parameters, credible intervals for predictions, principled noise estimation, and robustness against overfitting — while keeping your original symbolic model structure. 

The project exposes a scikit-learn style estimator **PostHocInference** that automates replacing numeric constants with learnable parameters, compiles the symbolic model into a Pyro probabilistic model, and runs MCMC sampling.

## Highlights
- Converts an equation given as SymPy expressions → Probabilsitic (Pyro) model. See [create_pyro_model()](post_hoc_inference/core/pyro_backend/sympy_to_pyro.py) in sympy_to_pyro.py
- Simple scikit-learn style API: PostHocInference(expr, input_symbols, output_dim, inference_params) with fit, predict, get_posterior, inference_diagnostics.
- Supports any parameterized function $f_\theta:(x_1, x_2, ..., x_N) \rightarrow (y_1, y_2, ..., y_M)$ with parameters $\theta \in \mathcal{R}^K$ and scalar inputs $x_i, y_j \in \mathcal{R} \text{, }\forall i, j$. 

Poit wise fit
![image](figures/eq_11_comp_26_loss_0.4980.png)

Probabilistic fit
![image](figures/eq11_posterior_predictive.png)


## Install project
### User Installation


To install the published package using `pip`, use the following command:

```bash
pip install mcl_post_hoc_inference --index-url "https://GIT_DEPLOY_TOKEN:glpat-NexIFAr779d7XfDfPcZznW86MQp1OmM3CA.01.0y0dx7lfc@gitlab.mcl.at/api/v4/projects/1113/packages/ pypi/simple" --upgrade
```

### Developer Installation

If you are a developer of this project or wish to contribute, the recommended setup is as follows:

- Clone this repository:

  ```bash
  git clone https://gitlab.mcl.at/c.findenig/post-hoc-inference.git
  cd Post_hoc_Inference
  ```

- Install the package in **editable mode**:

  ```bash
  pip install -e .
  ```

This approach ensures that any changes you make to the code or documentation are immediately applied when you run your code. Once satisfied with your modifications, commit them locally and utilize your Git expertise.

> **Note:** Direct pushing to the `main` branch is disabled.  
> Feel free to use your own branches or synchronize with the `develop` branch:
>
> ```bash
> git checkout develop
> ```
>
> Then create a merge request from your branch.


## Minimal Example
Minimal example usage of the tool - a more comprehensive example is provided in [showcase_post_hoc_inf.ipynb](notebooks/showcase_post_hoc_inf.ipynb). 

```python
import sympy as sp
import torch

from post_hoc_inference.core.post_hoc_inference import PostHocInference

# 1) Define a simple SymPy model: y = a*x + b
x = sp.symbols("x")
expr = 2. * x + 5.

# 2) Create synthetic data (batch, features) and targets (batch, output_dim)
X = torch.randn((4,1))   # shape (4,1)
y = torch.randn((4,1))    # shape (4,1)

# 3) Init estimator and run MCMC (small samples for demo)
inference_params = {
   "n_samples": 200, 
   "n_warmup_samples": 100, 
   "n_chains": 1, 
   "jit_compile": False
}
model = PostHocInference(expr, input_symbols=[x], output_dim=1, inference_params=inference_params)

# 4) Fit and predict
model.fit(X, y)                       # runs MCMC
preds = model.predict(X, n_predictive_samples=200)  # predictive samples 

# 5) Inspect posterior / diagnostics
idata = model.get_posterior()        # ArviZ InferenceData
diag = model.inference_diagnostics() # MCMC diagnostics

```

### Notes
- y must be 2D (batch, output_dim) and X (if present) must be 2D (batch, features).
- The compiled Pyro model uses Normal priors for parameters and a HalfNormal prior for observation noise; see [sympy_to_pyro.py](post_hoc_inference/core/pyro_backend/sympy_to_pyro.py) for details. 



## Available Commands

### `scanner`

The `scanner` command analyzes your project's dependencies and generates reports on licenses and vulnerabilities. If no file is specified, it defaults to using the `requirements.txt` file located in the root directory.

**Usage:**

```bash
scanner [path/to/your/requirements.txt]
```

Replace `path/to/your/requirements.txt` with the actual path to your `requirements.txt` file if it's not in the root directory.

> **Note:** If no file is specified, the command will automatically use `requirements.txt` from the root directory.

**About deps.dev:**

[deps.dev](https://deps.dev/) is a service developed by Google that assists developers in understanding the structure, composition, and security of open-source software packages. It provides insights into dependencies, licenses, security risks, and other critical health and security aspects for over 50 million open-source package versions. The data is sourced from various platforms, including package registries, the Open Source Vulnerability database, and code hosts like GitHub and GitLab. For more information, visit [https://deps.dev/](https://deps.dev/).

### `package`

The `package` command builds your Python package, ensuring it's ready for distribution. This process generates two files: a source archive (`.tar.gz`) and a built distribution (`.whl`), both located in the `dist` directory.

**Usage:**

```bash
package
```

This command will:
1. Check for the necessary dependencies (`setuptools`, `build`, `twine`).
2. Build the package using the configurations specified in your `pyproject.toml`.

If any required dependencies are missing, the command will prompt you to install them.

For more information on publishing packages, refer to the [GitLab Package Registry documentation](https://docs.gitlab.com/ee/user/packages/package_registry/).

## Publishing the Package

### Manually Publishing Packages

If you prefer not to run this process automatically, you can manually publish the package by following these steps:

1. **Install Dependencies**:

   Install the necessary dependencies with:

   ```bash
   pip install setuptools build twine
   ```

   or

   ```bash
   conda install setuptools build twine
   ```

2. **Build Your Package**:

   Execute the following command at the root of your repository:

   ```bash
   package
   ```

   This will create a folder named `dist` containing a `.tar.gz` and a `.whl` file.

3. **Publish Your Package**:

   Upload your package using Twine:

   ```bash
   - python -m twine upload --repository-url "https://gitlab.mcl.at/api/v4/projects/1113/packages/pypi" dist/* --verbose
   ```

   When prompted, enter your username and access token with scope 'api' to verify that you have permissions to upload to the repository.

   After that, users can install your package as described in Section [User Installation](#user-installation)

4. **Cleaning Up**:

   You can now safely delete the `dist` folder.

> **Note:** Alternatively, you can use the `build` command to only build the package (not recommended):
>
> ```bash
> python -m build
> ```

## Project History & Documentation

This project maintains **two structured logs** to document its evolution:

1. **Changelog (`CHANGELOG.md`)** – Tracks code changes, feature additions, and bug fixes in a structured release format.
2. **Engineering Diary (`ENGINEERING_DIARY.md`)** – Captures technical decisions, debugging insights, and research notes.

Depending on the project's needs, you may choose to maintain one or both logs.

## Repository Creation Info

This repository was created using the [ECML Data-Science Project Template](https://gitlab.mcl.at/ecml/generic-team-repos/data-science-project-template), which implements best practices for software development, auto-documentation, pip-installability, and additional boilerplate code to assist in starting a professional codebase for **Python** projects, primarily with a **data-driven** design in mind.

This project was created with **[ECML Data-Science Project Template](https://gitlab.mcl.at/ecml/generic-team-repos/data-science-project-template) version 0.9.6.dev1+g942b92f**.

---

This README provides a comprehensive overview of using, publishing, and analyzing Python packages, along with specific information about **deps.dev**.

