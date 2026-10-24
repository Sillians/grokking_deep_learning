### **Problem Analysis**

1. You installed `numpy` with `!pip3 install numpy` inside the notebook, but your project is managed with **Poetry** (`pyproject.toml` + `poetry.lock`).

   * `!pip install` installs into the environment that Jupyter sees, which may **not** be the same Python interpreter Poetry or VS Code is using.
   * VS Code notebooks can sometimes default to the global Python interpreter rather than the Poetry venv.

2. In your `pyproject.toml`, `numpy` is under **dev dependencies**:

   ```toml
   [tool.poetry.group.dev.dependencies]
   numpy = "^1.23.5"
   ```

   That means `numpy` is only installed when you install **dev dependencies** (`poetry install --with dev`), but not necessarily available if you did a plain `poetry install`.

---

### ✅ **Steps to Fix**

1. **Activate the Poetry environment**:

   ```bash
   poetry shell
   ```

   This ensures you’re inside the venv Poetry manages.

2. **Install all dependencies including dev**:

   ```bash
   poetry install --with dev
   ```

   This will install `numpy`, `torch`, `pandas`, etc.

3. **Verify numpy is installed**:

   ```bash
   poetry run python -m pip show numpy
   ```

   You should see version `1.23.5` (or higher, within spec).

4. **Ensure VS Code is using the Poetry interpreter**:

   * Open Command Palette → **Python: Select Interpreter**
   * Pick the one inside `.venv` created by Poetry (it will look like `.venv/bin/python` or `.venv\Scripts\python.exe`).
   * Restart the Jupyter kernel.

5. **Alternative check inside notebook**:
   Add this at the top of your notebook:

   ```python
   import sys
   print(sys.executable)
   ```

   Make sure it points to the Poetry venv, not system Python.

---

⚡ If you still want to use `!pip install ...` in your notebook, you should bind it to Poetry’s interpreter like:

```python
!{sys.executable} -m pip install numpy
```

This ensures the package goes into the correct environment.

---
