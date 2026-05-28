# Welcome to the Ant Foraging Seminar Wiki!

This Wiki is your central hub for successfully navigating the repository and completing the daily exercises.

## 📅 Navigation (Seminar Days)
- **Day 0:** Introduction to Agent-Based Modeling and the Mesa Framework.
- **Day 1:** Exploring Correlated Random Walks.
- **Day 2:** Central Place Foraging and returning to the nest.
- **Day 3:** Stigmergy and Pheromone-based communication.

---

## 🛠️ Environment & Tools Glossary

Before you start writing code, it's essential to understand the tools we use:

### IDE (Integrated Development Environment)
An IDE is a software application that provides a comprehensive environment for writing code (e.g., VS Code, PyCharm, Thonny). It features syntax highlighting, autocompletion, debugging, and file management, making it much easier to manage complex projects than a simple text editor.

### Virtual Environment (venv)
A virtual environment is an isolated workspace for your Python projects. It ensures that the specific versions of packages (like `mesa` or `numpy`) you install for this seminar do not conflict with other projects on your computer. Always make sure your virtual environment is activated before installing packages or running your code!

### Jupyter Notebook
Jupyter Notebooks (`.ipynb` files) are interactive documents where you can combine live, executable Python code with formatted text (Markdown), equations, and visualizations. We use them heavily for exercises because you can run your models and instantly see plots and outputs inline.

---

## 🐍 Python Basics Refresher

To ensure you can follow along with the repository code and successfully navigate the seminar, we have compiled a detailed refresher covering the most critical Python concepts. 
We strongly encourage referencing the [Official Python Documentation](https://docs.python.org/3/) and the highly recommended **[Scientific Python Lectures](https://lectures.scientific-python.org/)** for deeper dives into these topics. The Scientific Python Lectures are explicitly tailored for research and modeling!

### 1. Control Flow & Indentation
Unlike other languages that use brackets `{}`, Python relies on whitespace indentation to define code blocks (e.g., inside `if` statements or functions).
```python
x = 5
if x > 0:
    sign = "positive" # Indented block
print(sign)           # Outside the block
```
If you forget to indent, Python will throw an `IndentationError`. *(See the [Scientific Python Lectures on Control Flow](https://lectures.scientific-python.org/intro/language/control_flow.html) for more examples).*

### 2. Mathematics: True vs. Integer Division
When doing math, understand the difference between `/` and `//`:
- `7 / 2` yields `3.5` (True division, returns a float)
- `7 // 2` yields `3` (Integer/Floor division, returns an int)

### 3. Lists, Indexing, & Mutability
Lists are 0-indexed. Accessing an index that doesn't exist raises an `IndexError`.
```python
pets = ["cat", "dog", "python"]
print(pets[0]) # Output: "cat"
```
**Mutability:** Lists are *mutable* (you can change their elements). Strings are *immutable*.
```python
pets[0] = "bat" # Works!
s = "abc"
# s[0] = "x"    # Raises TypeError! Strings cannot be mutated in place.
```

### 4. For Loops & Ranges
The `range(n)` function generates numbers from `0` to `n-1`.
```python
for i in range(3):
    print(i) # Prints 0, 1, 2
```

### 5. Iterable Unpacking
You can easily "unpack" lists or ranges into variables. The `*` operator gathers remaining items into a list.
```python
a, *b, c = range(5)
# a = 0
# b = [1, 2, 3]
# c = 4
```

### 6. NumPy: Views vs. Copies & Broadcasting
When working with arrays, it's critical to know when you are editing a *copy* of data versus the *original* data.
- **Slicing creates a view:** `b = a[1:]`. Changing `b` *will* change `a`.
- **Boolean indexing creates a copy:** `b = a[a > 1]`. Changing `b` *will not* change `a`.

**Broadcasting:** NumPy automatically aligns shapes for arithmetic. If you add a shape `(3,)` array to a shape `(3, 1)` array, the result broadcasts to `(3, 3)`.

> 💡 **Highly Recommended:** For an excellent visual deep dive into broadcasting and views vs copies, please study the **[Scientific Python Lectures on NumPy](https://lectures.scientific-python.org/intro/numpy/index.html)**.

### 7. Object-Oriented Programming (OOP)
Our ABM models are built using classes. **For a deep dive into OOP, check out [Scientific Python Section on Object-Oriented Programming](https://lectures.scientific-python.org/intro/language/oop.html).**

#### Class vs. Instance Attributes
- **Class Attribute:** Shared by all instances. Defined outside `__init__`.
- **Instance Attribute:** Unique to the instance. Defined inside `__init__` using `self`.
If an instance attribute is deleted (`del c.state`), Python falls back to reading the class attribute!

#### Inheritance & MRO (Method Resolution Order)
Classes can inherit from multiple parents (`class Child(Left, Right):`). Python searches for methods from left to right. If both `Left` and `Right` have a `who()` method, `Child` will use the one from `Left`.

#### Properties (`@property` decorator)
A decorator like `@property` allows you to define a method that can be accessed like an attribute.
```python
class Parent:
    def __init__(self):
        self._x = 1
        
    @property
    def x(self):
        return self._x

p = Parent()
print(p.x) # Accessed without parentheses!
```
*Warning:* If a child class redefines `x = 2` as a plain class attribute, it overrides the property!

### 8. Custom Python Modules (Writing & Importing)
As your models grow, you shouldn't keep all your code in a single Jupyter Notebook. You can write your own Python module (a `.py` file) and import it.

For example, in `Day 2`, the ant logic is inside `model.py`.
In your Jupyter Notebook, you can import it like this:
```python
from model import AntModel, AntAgent

# Now you can use the classes defined in your .py file!
my_model = AntModel(...)
```
*Note: If you modify your `.py` file while the Jupyter Notebook is running, you may need to restart the notebook kernel, or use the `%autoreload` magic extension to see the changes.*

---

## 📝 Note for Maintainers: Updating the Cookbook

If you ever add new examples to `wiki/Cookbook.ipynb` and want to update the `Cookbook.md` file displayed in the Wiki (complete with the executed plots and outputs), you can do so by running the following command in your terminal while your virtual environment is activated:

```bash
jupyter nbconvert --execute --to markdown wiki/Cookbook.ipynb
```
*This command executes all the cells from top to bottom and exports the results directly into the Markdown file and an associated images folder.*
