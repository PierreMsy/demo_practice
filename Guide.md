# Python guidelines

## 🛠️Tools

### Linter / code formater: Ruff

Having consistent code style is essential when working on a shared codebase. Automating formatting and fixing simple, common flaws can save a lot of time. 

📐 **What it does:**

Ruff is a very fast Python linter and code formatter. It enforces pre-defined rules and style on the code by highlighting violations (linter) and automaticaly fix some of thoses (code formatter).
It replaces the usual suite of tools like `flake8`, `isort`, `pyupgrade` and `black`.
Can be integrated in a CI/CD pipeline (continuous integration, continuous delivery).

**How to use it:**

Running from the command line:

- Check for errors: `ruff check --preview .`
- Fix errors automatically: `ruff check --fix .`
- Format your code: `ruff format .`

Integration with your IDE (if it happen to be VS code):

1. Install the ruff extension ([link](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)) 
2. Add `"ruff.format.preview"`: true to your `settings.json` to benefit from the linter.
3. Optionally, add shortcuts to `editor.action.formatSelection`, `ruff.executeAutofix,` `ruff.executeOrganizeImports` and `ruff.executeFormat` to benefit of the code formatting.

Ruff explained [documentation](https://docs.astral.sh/ruff/).

---

### Documentation: Sphinx

Having a clear, navigable, and up to date documentation is a challenge in a code project. Sphinx helps you do so by automating what can be automated at the cost of writing a few `.rst` files.

📝 **What it does:**

Sphinx is a documentation generator that converts reStructuredTxet files to different output formats (HTML, PDF and more). It pulls documentation from your code’s docstring, ensuring the documentation synchronized with the code. It is the standard for Python projects.
It relies on `.rst` files (reStructuredText) where you structure your documentation and use `autodoc` to pull information from your python docstring. It allows you to create tables and cross references to different parts of your documentation.

**How to use it:**

1. Setup Sphinx

```bash
pip install sphinx # install Sphinx
cd ./docs          # go to your project docs folder
sphinx-quickstart  # run the interactive setup wizard to create necessary configuration files and directory structure.
```

1. Write your documentation ins `.rst` files

```bash
.. automodule:: my_project.my_module
   :members:
```

1. Build the doc using the `make` command.

In your docs folder, running `make html` will generate your documentation in html format in the `_build/html` directory.

Sphinx [documentation](https://www.sphinx-doc.org/en/master/). 

<aside>
💡

**Alternative: Pydoc for a lightweight documentation** 
[Pydoc](https://docs.python.org/3/library/pydoc.html) is Python’s built-in documentation system. It also generates documentation directly from docstrings, without any configuration files, `.rst` files, or build steps.
Useful when you open a legacy project and you want a quick doc to inspect the project.

```jsx
python -m pydoc my_project.my_module
python -m pydoc -w my_project   # generates HTML files
```

</aside>

---

### Integrated development environment: Visual Studio Code

Browsing a large codebase and using development utilities such as a debugger or Git integration requires a dedicated solution. 

🪟**What it does:**

VS Code integrates most development tools into a single solution, boosting productivity with its rich ecosystem of extensions.
It provides IntelliSense (smart code completion), a powerful debugger, an integrated terminal, and seamless Git integration.
It is a highly configurable IDE.

**How to use it:**

1. Download and install VS Code from the [official website](https://code.visualstudio.com/).
2. Install the Python extension in VS Code. From the Extensions view search for Python and install the official extension to get features like linting, debugging, and intelliSense.
3. Select your Python interpreter. From the Command Palette (Ctrl+Shift+P) type “Python: Select Interpreter” and choose the Python executable from the Conda environment you're working in. 
4. Optional, customize VS Code by editing the `settings.json` file.

**Some useful extensions:**

- **Python:** Essential starting point for all Python development.
- **Ruff**: Integrates the Ruff linter and formatter directly into the editor.
- **Jupyter**: Work with Jupyter notebooks / lab directly inside VS Code.
- **Pylance**: Improves Python code completion.
- **GitLens**: Improves Git integration allowing you to easily identify code ownership and see history.
- **Better Comments**: Creates more expressive comments by introducing special kind such as TODOs.
- **Docker**: Essential for building, managing, and deploying containerized applications in VS Code.
- **Python Docstring Generator**: Quickly generate a template of docstrings for python functions.
- **Rainbow CSV**: Improve readability of (small) .csv files (comma separated values).

<aside>
⚠️

**Only install only extensions that will provide you real value; too many extensions can slow down VS code, so it is a trade-off**

</aside>

---

### Test: Pytest

Code can break in unexpected ways when you make changes. Pytest ensures your code works correctly by running automated checks, preventing regressions, and giving you the confidence to refactor and add new features without introducing new bugs.

🔍**What it does:**

Pytest is a popular testing framework that is both simple and scalable. It provides detailed output on test failures, and a rich ecosystem of decorators (fixtures, mocks, parametrization) to make testing easier. It is based on simple assert statements for checking conditions and uses simple naming patterns to for test discovery.
It can be integrated in a CICD pipeline.

**How to use it:**

1. Install Pytest in your Python environment: `pip install pytest`.
2. If you use VS code, install the Python Test Explorer for Visual Studio Code ([here](https://marketplace.visualstudio.com/items?itemName=hbenl.vscode-test-explorer)).
3. (optional: create a `tests` folder next to your package folder) Create a Python file with a name starting by test (e.g `test_my_module.py`). Inside create a function that starts with `test_` with an assert statement as in the example below.
4. Use your test extension to discover the tests (parse the solution to find the pattern mention above) so you can easily run your tests. 

```python
# content of my_module.py you want to test
def add(x, y):
    return x + y
    
# content of test_my_module.py
from my_module import add

def test_add_method_sum_arguments():
    assert add(2,3) == 5
```

**Tips:**

- **Unit tests** that test a single functionality and have a clear intent and name are more informative than complex functional tests.
- **Test fixtures** are a convenient way of creating objects for multiple tests (for cleaner and more efficient tests). To use them:
    - Define a function that returns some object.
    - Use the `@pytest.fixture` decorator on that function to define it as a “fixture”.
    - Use the function name as an argument in your tests.
- **Test parameterization** is convenient way of testing multiple cases from a single test implementation. To do so:
    - Use the decorator : `@pytest.mark.parametrize(args_name, [(val_args_1), … (val_args_n)])`
    - Receive parameterized args as method args in your test function.
- **Test mocking** is convenient way of simulating behaviors from third party objects. To use mocks:
    - Use `unittest.mock` (standard library) or pip install `pytest-mock`.
    - Use the`@patch` decorator with target function as argument, as in `@path(rdkit.Chem.MolToSmiles)`.
    - You get a “MagicMock” object as first argument of your test. You can use the function `side_effect` on those objects to define their behaviors.

---

### Packages manager: Poetry

Managing project dependencies can lead to the "it works on my machine" problem. Poetry solves this by using a lock file to guarantee that every developer uses the exact same versions of libraries, ensuring consistent and reproducible builds.

📦**What it does:**

Poetry is a modern tool for Python dependency management and packaging.
It manage a specific project's Python package dependencies from a single `pyproject.toml` configuration file, ensuring reproducible builds with its poetry.lock file.
It is more efficient at solving package dependencies than the `requirement.txt` + `setup.py` combo.

**How to use it:**

1. install poetry: `pipx install poetry` or follow the [poetry doc](https://python-poetry.org/docs) for more details.
2. Initialize poetry:
    1. In a new project, use `poetry new my-new-project`, that will create a poetry config `pyproject.toml`, a `README.md`, a `tests` folder and a `my_new_project` folder containing a basic `init.py`.
    2. In an existing project, use `poetry init` in your project folder. It will launch a wizard that will ask question to create the `pyproject.toml`.
3. Use Poetry to add dependency to your project.
    1. Add a dependency to your package `poetry add any_library`. It will add the dependency to `pyproject.toml` and update `poetry.lock` and optionally install the package to your virtual environment.
    2. Add development-only library (like pytest) to your package `poetry add any_library --group dev`.
4. Install all dependencies of a project `poetry install`. 
    
    <aside>
    💡
    
    If you do not want to use virtualenvs (but rather yout conda env for instance) you can use the command `poetry config virtualenvs.create false --local` .
    
    </aside>
    
5. Optionally (not advised if you also use conda), run your code with `poetry run python your_script.py`. Poetry automatically creates and manages an isolated virtual environment for your project.

Alternative: https://github.com/astral-sh/uv (convenient if you have to often build and share docker images).

<aside>
💡

**Editable install**: Poetry already handle the current local project it manage as an editable installs, changes in the code will be taken into account. That is the behavior you get with pip in developer mode `pip install -e .`.
If you want to install *another* local package in editable mode (equivalent to `pip install -e ../my-other-library`) you have to use the command `poetry add -e ../my-other-library`. This will add the local package as a *path dependency* in your `pyproject.toml` .

</aside>

---

### Environment manager: Conda.

Different projects often require conflicting versions of Python or other libraries. Conda solves this by creating completely isolated environments for each project, so their dependencies never interfere with each other.

🐍 **What it does:**

Conda allows you to create isolated environments containing specific versions of Python and packages, so that different projects don't interfere with each other.
It is adapted to data science because it can manage non-Python packages (like CUDA or C++ libraries).

<aside>
👉

**Example**: you want to create a project with tensorflow and a specific version of python. 
with **pip**:  
- Install the specific python version you need manually.
- create an environment `python -m venv my-project` and `source my-project/bin/activate`.
- Consult TensorFlow documentation to identify which NVIDIA driver, CUDA toolkit, and cuDNN versions you need.
- Download from the NVIDIA website and install the correct CUDA and cuDNN libraries, set the system paths correctly.
with **conda**: 
- `conda create -n my-project python=3.11 tensorflow` 
→ Conda identify all dependencies, it installs Python 3.11, the tensorflow Python package, and the specific  CUDA toolkit and cuDNN libraries from NVIDIA that are *guaranteed* to be compatible with that version of TensorFlow.

</aside>

<aside>
💡

Alongside the defaults “channel” curated by Anaconda .Inc, the remote server where packages are stored, conda make available a open source community driven repository, **conda-forge**. It includes thousands of packages, some you cannot find in PyPi because not in python.

You can explicitly target this repository with the command `conda install -c conda-forge some-package` but the best practice is to set a priority to avoid mixing source from the two main channels with `conda config --add channels conda-forge` and `conda config --set channel_priority strict`.

</aside>

**How to use it:**

1. Create and activate an environment with a specific python version: `conda create --name myenv python=3.10` and `conda activate myenv`
2. Install and list installed packages: `conda install numpy pandas`, `conda list`
3. Deactivate and remove an environment (and all its installed packages): `conda deactivate`, `conda env remove —name myenv` 

<aside>
⚠️

**Pip** (with `venv`) **is the simpler**, more lightweight, and standard tool for development that does not integrate in a complex dependencies environment (such as deep learning).  
Conda's powerful solver might take longer to calculate the environment, and the environment itself will include the full conda infrastructure.
Also, pip install are often faster because it relies on wheels (already compiled Python code) instead of downloading package as *tar.gz.* Conda does not because of its language agnostic philosophy.

</aside>

<aside>
🚀

**Mamba: A faster replacement for Conda**
Mamba is a modern reimplementation of Conda’s dependency solver written in C++. It uses the same commands, the same environments, and the same channels (including conda-forge) — but solves environments orders of magnitude faster. Because some large package installation just take too much time with conda.

```bash
conda install -c conda-forge mamba
mamba create -n myenv python=3.11 pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
mamba install numpy pandas
```

</aside>

---

### Deep learning: Pytorch lightning.

Writing raw deep learning code can be repetitive, tedious and mixed with low-level infra considerations (cpu, gpu, saving, checkpoints). PyTorch Lightning organizes your code to separate the science from the engineering.

🎛️ **What it does:** 

PyTorch Lightning is a lightweight wrapper for PyTorch. It structures your deep learning code by decoupling the research logic (model architecture, loss calculation, optimization) from the hardware logic (GPU distribution, precision, checkpointing).
It creates a standard template for models, making them easier to share, reproduce, scale and industrialize.

**How to use it:**

1. Install the library: `pip install pytorch-lightning`.
2.  Refactor or build your PyTorch model code into a `LightningModule`.  Encapsulate the core logic into specific PyTorch lightning methods (`training_step`, `configure_optimizers`, etc.). Create `DataModule` and `Dataloader`.
→ comprehensive [step by step guide](https://github.com/Lightning-AI/pytorch-lightning/blob/master/docs/source-pytorch/_static/images/general/pl_quick_start_full_compressed.gif) on how to refactor raw PyTorch code into Pytorch Lightning code.
3. Use the `Trainer` object to handle the execution.

```python
# Example of fiting a DL model with the pytorch litghning trainer and configuration of low level settings.a
trainer = L.Trainer(
	max_epochs=10,
	accelerator="auto", # what kind of hardware to use ("cpu", "gpu", "tpu", ...)
	devices="auto", # how many (or which specific) units of the accelerator to use.
	precision="16-mixed",
	strategy="ddp" # Distributed Data Parallel
)
model = MyModel()
trainer.fit(model, train_dataloaders=train_loader)
```

PyTorch lightning [clear guides](https://lightning.ai/docs/pytorch/stable/).

---

### Configuration & Validation: Pydantic.

Configuration driven code is key to industrialize modern ML code but using plain dictionaries or loading raw json/yaml files is error-prone.
Pydantic solves this by enforcing type safety and validation at the moment the configuration is loaded.

⛩️ **What it does:** 

Pydantic is a data validation library that uses Python type hints. It is more than just a typed dataclass but rather a data parsing and transformation tool.
For configuration, it allows you to define your settings as a Python class. When you instantiate the class (from environment variables or a file for instance), Pydantic validates that the data matches the expected types. If a value is missing or of the wrong type (e.g., a string instead of an integer), it creates descriptive error immediately, preventing runtime crashes later in your pipeline.

**How to use it:**

1. Install Pydantic (and pydantic-settings for config management): `pip install pydantic pydantic-settings`. (`pydantic-settings`  is an optional extension to hjanfle environnement variables `.env`) 
2. Define your configuration class by inheriting from `BaseModel` (for data) or `BaseSettings` (for environment variables/configs).
3. Use standard Python type hints to define fields, give defaults and use Pydantic decorator to create data validation class.

```python
# Example of a configuraiton class for a deep learning class.
class DeepPredBlockConfig(BaseModel):
    """Configuration of the DeepPredBlock."""
    objective: Objective = Field(default=Objective.REGRESSION)
    checkpoints_path: Path = Field(
        default=global_cfg.path.model,
        description="Directory where model weights will be saved."
    )
    checkpoint_prefix: str = Field(
        default="",
        description="String to append at the begining of checkpoints."
    )
    default_max_epochs: int = Field(default=3, gt=0)
    accelerator: str = Field(default="auto")
    devices: int | str | list = 1
    trainer_kwargs: dict[str, Any] | None = Field(default=None)

    eval_metric: str = "val_loss"
    eval_mode: str = "min"

    @field_validator('checkpoints_path')
    @classmethod
    def _resolve_path(cls, path: Path | str) -> Path:
        """Transform to path to an absoult path."""
        absolut_path = Path(path).resolve()
        return absolut_path

    @field_validator('checkpoint_prefix')
    @classmethod
    def _sanitize_prefix(cls, prefix: str) -> str:
        initial_prefix = prefix
        prefix = sanitize_name(prefix)
        if prefix != initial_prefix:
            logger.warning(
                f"Prefix {initial_prefix} rename to {prefix} "
                "for file stafety.")
        return prefix
```

**Benefits:**

- **Type Safety:** You get IDE support (autocomplete, linting) for your configuration object (e.g., `config.batch_size` will be recognized as an `int`).
- **Fail Fast:** If you accidentally set the learning rate to "low" (string) instead of `0.001` (float), the code crashes instantly with a clear error message, rather than silently failing during training.
- **Parsing:** It automatically casts types where possible (e.g., the string "32" in an env var becomes the integer `32` in your code) for serialization.

<aside>
💡

`autodoc_pydantic` extension can bridge the need of docstring of Sphinx and the Pydantic class model for configuration object.

</aside>

### Recap

- Enforce code style and auto-format /fix: 📐 **Ruff**
- Create modern and up-to-date doc: 📝 **Sphinx**
- Have a modern integrated development environnement: 🪟 **Visual Code**
- Rubostify your project, secure addition of new features:🔍 **Pyest**
- Manage the environment of your different projects: 🐍 **Conda**
- 📦**Poetry**

## 🥋Code guidelines

### **Typing**

In programming, typing is labeling your variables to specify what kind of data they are allowed to hold. Python’s typing is strong and dynamic. Strong because you cannot add a string and an integer without explicit casting, and dynamic because an object type is checked at runtime.
Because of its dynamic nature, you do not have to explicitly type object in your code as you would languages like C.

**Good practice**: To add robustness and have a more explicit code use type hint. Type hints enable static analysis type checking where your solution can be analyzed for type inconsistencies, catching potential bugs before runtime.

Below you can find a example of a function with typed argument and return value:

```python
def export_report(
    self,
    output_dir: Path | str,
    dataset_selection: Sequence[str] | None = None,
    one_fig_per_dataset: bool = False,
    suffix: str | None = None,
    image_extension: str = ".svg"
) -> None:
    """Exports the report into images and CSVs. [...]"""
```

**How activate static analysis in VS code:** Add `"python.analysis.typeCheckingMode": "basic"`, to your VS code `settings.json`

### Docstring

The Google docstring style is the standard because of its readablity and popularity.
The layout is described as follow:

```python
def my_funciton(arg_name: int) -> float:
"""Summary line: one-line summary of what the function does.

Detailed description: optional, more detailed description.

Args:
    arg_name (type): name, followed by its type in parentheses, a colon,
	      and then a description of the argument.
	      
Returns:
    type: type, a colon, and a description of the return object.
    
 Raises:
    ErrorType: when the error occurs.
"""
```

Google [docstring documentation](https://google.github.io/styleguide/pyguide.html#381-docstrings).

### Naming conventions

- PascalCase: `ClassName`, `ExceptionName`
- snake_case: `module_name`, `package_name`, `method_name`, `function_name`, `function_parameter_name`, `local_var_name`, `global_var_name`
- CAPITALIZED_SNAKE_CASE: `GLOBAL_CONSTANT_NAME`

**underscore convention:**

- `_single_leading_underscore`:  hint that a variable or method is intended for internal use only.
- `__double_leading_underscore`:  private method, prevent override by child class (name mangling).
- `__double_leading_and_trailing_underscore__`: convention for special “magic” functions like `__init__`, `__len__`, and `__str__` .
Take note that re-implementing magic functions can make your code more natural and “pythonic” but it is not advised to write your own magic functions.
- `trailing_underscore_`: machine learning convention for fitted objects (whose internal state has been adapte   d to data). It can also be used to avoild collision  with language reserved keywords such as “class” (by using “class_”).

### Development guidelines

This is a vast topic that cannot be exhaustively covered in a short document. Below are some general useful principles to keep in mind and also resources to delve into.

**SOLID Principles**: 

- **S**ingle responsibility principle: a class or a method should do one thing only. This makes your code easier to test, debug, and reuse.

```python
# Bad: One class with two responsibilities
class User:
		"""Manages user data and persistence."""
    def __init__(self, name: str):
        self.name = name

    def get_name(self) -> str:
        return self.name

		# This part violates SRP
    def save_to_database(self):
        #[...]
```

```python
# Good: Each class has a single responsibility
class User:
    """Manages user data."""
    def __init__(self, name: str):
        self.name = name

    def get_name(self) -> str:
        return self.name

class UserRepository:
    """Handles user persistence."""
    def save(self, user: User):
        print(f"Saving {user.get_name()} to the database...")
```

- **O**pen closed principle: Function and classes must be open for extension but closed for modification. Adding new functionalities should be easy and with minimal changes for the existing code.

```python
# Bad: Extension would mean modification
class Square:
    def __init__(self, side):
        self.side = side

class Circle:
    def __init__(self, radius):
        self.radius = radius

def calculate_area(shape):
    """Compute area for each type of shape""
    if isinstance(shape, Square):
        return shape.side ** 2
    elif isinstance(shape, Circle):
        return 3.14 * shape.radius ** 2
    # Adding a Triangle would mean modifying this function
```

```python
# Good: Open for extension, closed for modification
from abc import ABC, abstractmethod

class Shape(ABC):
    """Abstract base class for all shapes."""
    @abstractmethod
    def area(self) -> float:
        pass

class Square(Shape):
    def __init__(self, side):
        self.side = side
    
    def area(self) -> float:
        return self.side ** 2

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self) -> float:
        return 3.14 * self.radius**2

def print_area(shape: Shape):
    """Print area of a shape."""
    # This function is closed for modification.
    print(f"The area is: {shape.area()}")

# We can add a new shape without changing print_area
class Triangle(Shape):
    def __init__(self, base, height):
        self.base = base
        self.height = height
    
    def area(self) -> float:
        return 0.5 * self.base * self.height
```

- **L**iskov substitution principle: Subtypes must be substitutable for their base type without altering the correctness of a program. Child classes must not break the functionality of their base class.

```python
# Bad: Rectangle child class does not behave like a rectangle
class Rectangle:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
    
    def set_width(self, width: int):
        self.width = width
    
    def set_height(self, height: int):
        self.height = height
        
    def get_area(self) -> int:
        return self.width * self.height

class Square(Rectangle):

    def set_side(self, side: int)
        self.width = side
        self.height = side

    def set_width(self, width: int):
        self.set_side(width)

    def set_height(self, height: int):
		    self.set_side(height)
		    
def test_rectangle_resizing(rectangle: Rectangle):
	
	rectangle.set_width(2)
	rectangle.set_height(4)
	
	assert my_rectangle.get_area() == 8
		    
test_rectangle_resizing(Rectangle(1, 1)) # Ok
test_rectangle_resizing(Square(1, 1)) # Fail 16 =!=8
```

```python
# Good: both Shape childs behave like shapes
class Shape(ABC):
    @abstractmethod
    def get_area(self) -> int:
        pass
        
class Rectangle(Shape):
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
    
    def set_width(self, width: int):
        self.width = width
    
    def set_height(self, height: int):
        self.height = height
        
    def get_area(self) -> int:
        return self.width * self.height

class Square(Shape):
    def __init__(self, side: int):
        self.side = side

    def set_side(self, side: int):
        self.side = side
        
    def get_area(self) -> int:
        return self.side * self.side

# This would cause a clear type Error.
test_rectangle_resizing(Square(1))
```

- **I**nterface segregation principle: An object must not be forced to depend on methods it does not use. In python, abstract classes should be focused and minimal and you rather use multiple of them instead of one big one-size-fits-all abstract class.

```python
# Bad: Interface inducing several not linked dependencies
from abc import ABC, abstractmethod

class IWorker(ABC):
    @abstractmethod
    def work(self):
        pass
    
    @abstractmethod
    def eat(self): # not always relevant
        pass

class HumanWorker(IWorker):
    def work(self):
        [...]  # relevant
    
    def eat(self):
        [...]  # relevant

class RobotWorker(IWorker):
    def work(self):
        [...]  # relevant
    
    def eat(self):
        [...]  # Forced implementation that is irrelevant here
```

```python
# Good: Segregated interfaces: no irrelevant implementation
from abc import ABC, abstractmethod

class IWorkable(ABC):
    @abstractmethod
    def work(self):
        pass

class IEatable(ABC):
    @abstractmethod
    def eat(self):
        pass

class HumanWorker(IWorkable, IEatable):
    def work(self):
        [...]   # relevant
    
    def eat(self):
        [...]   # relevant

class RobotWorker(IWorkable):
    def work(self):
        [...]   # relevant
```

- **D**ependency inversion principle: High level module should not depend on low-level module. High level business logic should not be coupled tightly to low level-implementation (database / API for instance). The most important is not to mention infra related code in your high level (business) code.
Passing dependencies as arguments (dependency injection) is a common way to achieve this.

```python
# Bad: High-level module depends on low-level module
class MySQLDatabase: # A low-level module.
    def get_data(self) -> str:
        [...]

class ReportGenerator: # A high-level module.
    def __init__(self):
        self.db = MySQLDatabase() # Tight coupling 
    
    def generate_report(self):
        data = self.db.get_data()
        self.formalize_report(data)
      
     def formalize_report(self, data):
	       [...] 
	       
# Cannot easily generate a report from another data source.
```

```python
# Good: High-level module depend on an abstraction
from abc import ABC, abstractmethod

class IDataSource(ABC):
    @abstractmethod
    def get_data(self) -> str:
        [...]

class MySQLDatabase(IDataSource): # Fist low-level module.
    def get_data(self) -> str:
        [...]

class ApiDataSource(IDataSource): # Second low-level module.
    def get_data(self) -> str:
        [...]

# High-level module: depends only on the abstraction.
class ReportGenerator:
    def __init__(self, data_source: IDataSource):
	      # Dependency is "injected"
        self.data_source = data_source 
    
    def generate_report(self):
        data = self.data_source.get_data()
        self.formalize_report(data)

# Can easily generate a report from another data source.
report1 = ReportGenerator(MySQLDatabase()).generate_report()
report2 = ReportGenerator(ApiDataSource()).generate_report()
```

**See also:**

- [Zen of python](https://peps.python.org/pep-0020/): 20 rules that convey the philosophy of the language.
- [Google python style guide](https://google.github.io/styleguide/pyguide.html): complete and comprehensive guide that explains the standards of code at Google.
- Comprehensive [design pattern guide](https://refactoring.guru/design-patterns/python): templates to solve common programming issues with python code example and explanations.

## Configurations

### Ruff configuration

Example of Ruff configuration: 

```toml
# ruff.toml
line-length = 80

[lint]
select = [
    "F",    # Pyflakes
    "E",    # Pycodestyle
    "N",    # pep8-naming
    "D",    # pydocstyle
    "UP",   # pyupgrade
    "I001", # isort
    "S",    # flake8-bandits
    "C4",   # flake8-comprehensions
    "SIM",  # flake8-simplify
    "W"
]
ignore = [
    "D105", # docstring required for magic methods
    "N803", # invalid argument name
    "N806", # non lowercase variable in function
    "D107", # docstring in __init__ (arguments should be in class docstring)
    "D100", # undocumented public module (usually redundant with class docstring)
    # Specific ignore for tests
    "S101",  # Use of asserts
    "D100",  # Docstring in public module
    "D101",  # Docstring in class
    "D102",  # Docstring in public method
    "D103",  # Docstring in public function
    "D104",  # Docstring in public package
    "D107",  # Docstring in __init__ of class
    "S403",  # pickle is insecure
    "S301",  # loading pickle is insecure
    "D200",  # One line summary.
    "F811",  # Redefinition of unused variable (issues with pytest fixtures)
]

[lint.pydocstyle]
convention = "google"

[lint.extend-per-file-ignores]
"__init__.py" = ["F401"]
```

### Sphinx configuration

ML Team Sirius project’s configuration:

```python
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Paths & project metadata
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

project: str = "Toy Binary Classification"
author: str = "Your Name"
year: int = datetime.now().year
copyright: str = f"{year}, {author}"

# Optional: a short version and full version (e.g. from your package)
# Try to import your package to get __version__ if available
try:
    from toy_binary_classification import __version__ as release  # type: ignore[import-not-found]
except Exception:
    release = "0.1.0"
version = release

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------

extensions: List[str] = [
    # Core
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    # Type hints
    "sphinx_autodoc_typehints",
]

templates_path: List[str] = ["_templates"]
exclude_patterns: List[str] = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# The master toctree document (Sphinx <7 calls this; in >=7 it's `root_doc`)
master_doc: str = "index"
root_doc: str = "index"

# ---------------------------------------------------------------------------
# HTML options
# ---------------------------------------------------------------------------

html_theme: str = "sphinx_rtd_theme"
html_static_path: List[str] = ["_static"]

# Optional: some basic theme options
html_theme_options: Dict[str, Any] = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "titles_only": False,
}

# ---------------------------------------------------------------------------
# Autodoc / autosummary / type hints
# ---------------------------------------------------------------------------

autodoc_typehints: str = "description"  # your original choice
autodoc_member_order: str = "groupwise"  # or "bysource"

autosummary_generate: bool = True
autosummary_imported_members: bool = True

# Put type hints in the description instead of signature
typehints_fully_qualified: bool = False
typehints_use_signature: bool = False

# ---------------------------------------------------------------------------
# Napoleon (Google / NumPy docstrings)
# ---------------------------------------------------------------------------

napoleon_google_docstring: bool = True
napoleon_numpy_docstring: bool = False
napoleon_include_init_with_doc: bool = True
napoleon_include_private_with_doc: bool = False
napoleon_use_param: bool = True
napoleon_use_rtype: bool = True

# ---------------------------------------------------------------------------
# Intersphinx (links to external docs)
# ---------------------------------------------------------------------------

intersphinx_mapping: Dict[str, tuple[str, str | None]] = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
}

# ---------------------------------------------------------------------------
# Misc settings and nicer defaults
# ---------------------------------------------------------------------------

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_css_files: List[str] = [
    # "css/custom.css",
]

# If you want to configure logging / warnings:
suppress_warnings: List[str] = [
    "autodoc.import_object",  # often noisy in early stages
]

# Example: use environment variable to tweak behavior in CI
on_rtd: bool = os.environ.get("READTHEDOCS", "") == "True"
if on_rtd:
    # E.g. disable autosummary in RTD if it causes issues
    autosummary_generate = True

```

### Poetry: pyproject.toml

ML Team Sirius pyproject.toml:

```toml
[tool.poetry]
name = "simple-lightning-classifier"  # distribution name on PyPI
version = "0.1.0"
description = "Toy binary classification demo package"
authors = ["You name <yourname@whatever.com>"]
readme = "README.md"
license = "MIT"
packages = [
    { include = "simple_lightning_classifier", from = "src" },
]

[tool.poetry.dependencies]
python = ">=3.12,<4.0"
numpy = ">=2.3.5,<3.0.0"
lightning = ">=2.6.0,<3.0.0"
scikit-learn = ">=1.7.2,<2.0.0"
pydantic = "^2.7"
pydantic-settings = "^2.12.0"

[tool.poetry.group.dev.dependencies]
ipykernel = "^6.0"
sphinx = "^7.0"
sphinx-autodoc-typehints = "^2.0"
sphinx-rtd-theme = "^2.0"
ruff = "^0.14.7"
pytest = "^9.0.1"

[tool.ruff]
line-length = 88
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
ignore = ["E501"]

[build-system]
requires = ["poetry-core>=2.0.0,<3.0.0"]
build-backend = "poetry.core.masonry.api"

```

---

## **Lexicon**

**CICD**: continuous integration and continuous delivery/deployment

**IDE**: Integrated development environment

**Module**: a single python file (`*.py`)

**Package**: a way to structure related modules together. It is a directory that contains python module and `__init__.py` files.

**Library**: general term for a collection of packages and modules that provides functionality to solve a specific set of problems.

`folder`

`file`

`code`