import genanki

cards = [
	# --- CS50P::BASICS ---
	(
		"""print("Hello, world!")""",
		"Outputs text.",
		"cs50p::basics",
	),
	(
		"""name = input("Name: ")""",
		"Reads user input.",
		"cs50p::basics",
	),
	(
		"""age = int("42")""",
		"String to integer.",
		"cs50p::basics",
	),
	(
		"""x, y = 1, 2""",
		"Tuple unpacking.",
		"cs50p::basics",
	),
	(
		"""a, b = b, a""",
		"Swap values.",
		"cs50p::basics",
	),
	(
		"""f"Hello {name}""",
		"f-string interpolation.",
		"cs50p::basics",
	),
	(
		"""print(1, 2, 3, sep=" | ")""",
		"Custom separator.",
		"cs50p::basics",
	),
	(
		"""print("no newline", end="")""",
		"Suppress newline.",
		"cs50p::basics",
	),
	(
		"""x = None""",
		"Null placeholder.",
		"cs50p::basics",
	),
	(
		"""bool("")""",
		"Empty string is False.",
		"cs50p::basics",
	),
	(
		"""bool([0])""",
		"Non-empty list is True.",
		"cs50p::basics",
	),
	(
		"""x is None""",
		"Identity test for None.",
		"cs50p::basics",
	),
	(
		"""isinstance(3, int)""",
		"Type membership check.",
		"cs50p::basics",
	),

	# --- CS50P::NUMBERS ---
	(
		"""2 ** 10""",
		"Exponentiation.",
		"cs50p::numbers",
	),
	(
		"""7 // 3""",
		"Floor division.",
		"cs50p::numbers",
	),
	(
		"""7 % 3""",
		"Modulo remainder.",
		"cs50p::numbers",
	),
	(
		"""divmod(7, 3)""",
		"Returns quotient, remainder.",
		"cs50p::numbers",
	),
	(
		"""import math
		math.sqrt(9)""",
		"Square root.",
		"cs50p::numbers",
	),
	(
		"""round(3.1415, 2)""",
		"Round to two decimals.",
		"cs50p::numbers",
	),
	(
		"""import random
		print(random.randint(1, 6))""",
		"Random int inclusive.",
		"cs50p::numbers",
	),
	(
		"""import random
		print(random.choice("abc"))""",
		"Random element.",
		"cs50p::numbers",
	),

	# --- CS50P::STRINGS ---
	(
		""""hello".upper()""",
		"Uppercase conversion.",
		"cs50p::strings",
	),
	(
		""""Hello".lower()""",
		"Lowercase conversion.",
		"cs50p::strings",
	),
	(
		"""" hi ".strip()""",
		"Trim whitespace.",
		"cs50p::strings",
	),
	(
		""""a,b,c".split(",")""",
		"Split to list.",
		"cs50p::strings",
	),
	(
		"""/".join(["a", "b"])""",
		"Join list.",
		"cs50p::strings",
	),
	(
		""""hello"[1:4]""",
		"Slice substring.",
		"cs50p::strings",
	),
	(
		""""hello"[::-1]""",
		"Reverse string.",
		"cs50p::strings",
	),
	(
		""""ß".casefold()""",
		"Aggressive lowercase.",
		"cs50p::strings",
	),
	(
		""""hi".encode()""",
		"str→bytes.",
		"cs50p::strings",
	),
	(
		"""b"hi".decode()""",
		"bytes→str.",
		"cs50p::strings",
	),
	# --- CS50P::CONDITIONALS ---
	(
		"""if x > 0:
		print("positive")""",
		"Basic if.",
		"cs50p::conditionals",
	),
	(
		"""if x > 0:
		print("pos")
	else:
		print("neg")""",
		"If-else.",
		"cs50p::conditionals",
	),
	(
		"""if 1 < x < 10:
		print("between")""",
		"Chained comparison.",
		"cs50p::conditionals",
	),
	(
		"""if x and y:
		print("both true")""",
		"Logical AND.",
		"cs50p::conditionals",
	),
	(
		"""if x or y:
		print("either true")""",
		"Logical OR.",
		"cs50p::conditionals",
	),
	(
		"""if not x:
		print("false")""",
		"Logical NOT.",
		"cs50p::conditionals",
	),
	(
		"""grade = "A" if score >= 90 else "B" """,
		"Ternary expression.",
		"cs50p::conditionals",
	),
	(
		"""if isinstance(x, int):
		print("integer")""",
		"Type check.",
		"cs50p::conditionals",
	),
	(
		"""if x is None:
		print("none")""",
		"Identity test.",
		"cs50p::conditionals",
	),

	# --- CS50P::LOOPS ---
	(
		"""for i in range(3):
		print(i)""",
		"Iterate 0-2.",
		"cs50p::loops",
	),
	(
		"""for char in "hi":
		print(char)""",
		"Iterate over string.",
		"cs50p::loops",
	),
	(
		"""for k, v in {"a": 1}.items():
		print(k, v)""",
		"Dict iteration.",
		"cs50p::loops",
	),
	(
		"""for i, v in enumerate(["a", "b"]):
		print(i, v)""",
		"Enumerate index + value.",
		"cs50p::loops",
	),
	(
		"""for i in range(0, 10, 2):
		print(i)""",
		"Range with step.",
		"cs50p::loops",
	),
	(
		"""while x < 5:
		x += 1""",
		"While loop increment.",
		"cs50p::loops",
	),
	(
		"""for i in range(5):
		if i == 3:
			continue""",
		"Skip iteration.",
		"cs50p::loops",
	),
	(
		"""for i in range(3):
		pass""",
		"Empty body.",
		"cs50p::loops",
	),
	(
		"""for i in range(3):
		print(i)
	else:
		print("done")""",
		"Loop else-clause.",
		"cs50p::loops",
	),
	(
		"""[x * 2 for x in range(3)]""",
		"List comprehension.",
		"cs50p::loops",
	),
	(
		"""[x for x in range(5) if x % 2 == 0]""",
		"Filtered comprehension.",
		"cs50p::loops",
	),
	(
		"""{x: x ** 2 for x in range(3)}""",
		"Dict comprehension.",
		"cs50p::loops",
	),
	(
		"""{x for x in "hello"}""",
		"Set comprehension.",
		"cs50p::loops",
	),
	(
		"""(x * 2 for x in range(3))""",
		"Generator expression.",
		"cs50p::loops",
	),
	(
		"""for i, j in zip(a, b):
		print(i, j)""",
		"Parallel iteration.",
		"cs50p::loops",
	),
	(
		"""for i in reversed(range(3)):
		print(i)""",
		"Reverse range.",
		"cs50p::loops",
	),
	(
		"""for i in sorted([3, 1, 2]):
		print(i)""",
		"Sorted iteration.",
		"cs50p::loops",
	),
	(
		"""for chunk in [data[i:i+2] for i in range(0, len(data), 2)]:
		print(chunk)""",
		"Chunked iteration.",
		"cs50p::loops",
	),
	(
		"""while queue:
		item = queue.pop(0)""",
		"Consume queue.",
		"cs50p::loops",
	),

	# --- CS50P::FUNCTIONS ---
	(
		"""def add(a, b):
		return a + b""",
		"Defines a function that adds.",
		"cs50p::functions",
	),
	(
		"""result = add(2, 3)""",
		"Calls function.",
		"cs50p::functions",
	),
	(
		"""def greet(name = "world"):
		return f"Hi {name}""",
		"Default parameter.",
		"cs50p::functions",
	),
	(
		"""def total(*args):
		return sum(args)""",
		"Variable positional arguments.",
		"cs50p::functions",
	),
	(
		"""def config(**kwargs):
		return kwargs.get("mode")""",
		"Variable keyword arguments.",
		"cs50p::functions",
	),
	(
		"""def outer():
		x = 0
		def inner():
			nonlocal x
			x += 1
			return x
		return inner""",
		"Closure with nonlocal.",
		"cs50p::functions",
	),
	(
		"""(lambda x: x ** 2)(3)""",
		"Lambda immediate call.",
		"cs50p::functions",
	),
	(
		"""sorted(data, key = lambda t: t[1])""",
		"Lambda as sort key.",
		"cs50p::functions",
	),
	(
		"""def gen():
		yield 1
		yield 2""",
		"Generator function.",
		"cs50p::functions",
	),
	(
		"""sum(x for x in range(5))""",
		"Generator expression sum.",
		"cs50p::functions",
	),
	(
		"""from functools import partial
		add5 = partial(int, base = 5)""",
		"Partial function application.",
		"cs50p::functions",
	),
	(
		"""def fib(n):
		a, b = 0, 1
		for _ in range(n):
			a, b = b, a + b
		return a""",
		"Iterative Fibonacci.",
		"cs50p::functions",
	),
	# --- CS50P::SCOPE ---
	(
		"""x = 1

def f():
	global x
	x = 2""",
		"Global variable write.",
		"cs50p::scope",
	),
	(
		"""def f():
		x = 0
		def g():
			nonlocal x
			x += 1
		g()
		return x""",
		"Nonlocal variable usage.",
		"cs50p::scope",
	),
	(
		"""locals()""",
		"Current local namespace.",
		"cs50p::scope",
	),
	(
		"""globals()""",
		"Global namespace dictionary.",
		"cs50p::scope",
	),

	# --- CS50P::DECORATORS ---
	(
		"""def deco(fn):
		def w(*a, **k):
			print(">")
			return fn(*a, **k)
		return w

	@deco
	def run():
		pass""",
		"Function decorator usage.",
		"cs50p::decorators",
	),
	(
		"""from functools import wraps""",
		"Preserve metadata when wrapping.",
		"cs50p::decorators",
	),
	(
		"""def memo(fn):
		cache = {}
		@wraps(fn)
		def w(x):
			if x in cache:
				return cache[x]
			cache[x] = fn(x)
			return cache[x]
		return w""",
		"Manual memoization decorator.",
		"cs50p::decorators",
	),
	(
		"""import functools as ft

	@ft.lru_cache(None)
	def fib(n):
		return n if n < 2 else fib(n-1) + fib(n-2)""",
		"LRU cache decorator.",
		"cs50p::decorators",
	),

	# --- CS50P::EXCEPTIONS ---
	(
		"""try:
		1 / 0
	except ZeroDivisionError:
		pass""",
		"Catch specific exception.",
		"cs50p::exceptions",
	),
	(
		"""try:
		risky()
	except (ValueError, TypeError):
		...""",
		"Multiple exception types.",
		"cs50p::exceptions",
	),
	(
		"""try:
		work()
	except Exception:
		pass
	else:
		print("OK")""",
		"Else runs if no exception.",
		"cs50p::exceptions",
	),
	(
		"""try:
		run()
	finally:
		cleanup()""",
		"Finally always executes.",
		"cs50p::exceptions",
	),
	(
		"""raise ValueError("bad")""",
		"Raise exception explicitly.",
		"cs50p::exceptions",
	),
	(
		"""class E(Exception): 
		pass
	raise E("oops")""",
		"Custom exception class.",
		"cs50p::exceptions",
	),
	(
		"""assert n > 0, "must be positive" """,
		"Assertion statement.",
		"cs50p::exceptions",
	),
	(
		"""try:
		...
	except Exception as e:
		print(type(e), e)""",
		"Access exception object.",
		"cs50p::exceptions",
	),

	# --- CS50P::FILEIO ---
	(
		"""with open("a.txt") as f:
		data = f.read()""",
		"Read text file.",
		"cs50p::fileio",
	),
	(
		"""with open("a.txt") as f:
		for line in f:
			pass""",
		"Iterate file lines.",
		"cs50p::fileio",
	),
	(
		"""with open("a.txt", "w") as f:
		f.write("x")""",
		"Write text file.",
		"cs50p::fileio",
	),
	(
		"""with open("a.txt", "a") as f:
		f.write("y")""",
		"Append file.",
		"cs50p::fileio",
	),
	(
		"""with open("b.bin", "rb") as f:
		f.read()""",
		"Read binary.",
		"cs50p::fileio",
	),
	(
		"""with open("b.bin", "wb") as f:
		f.write(b"\\x00")""",
		"Write binary.",
		"cs50p::fileio",
	),
	(
		"""from pathlib import Path
	Path("a.txt").read_text()""",
		"Pathlib read.",
		"cs50p::fileio",
	),
	(
		"""Path("a.txt").write_text("hi")""",
		"Pathlib write.",
		"cs50p::fileio",
	),
	(
		"""p = Path(".")
	[x for x in p.iterdir()]""",
		"List directory entries.",
		"cs50p::fileio",
	),
	(
		"""[str(p) for p in Path(".").rglob("*.py")]""",
		"Recursive glob search.",
		"cs50p::fileio",
	),
	(
		"""import os
	os.path.exists("a.txt")""",
		"Check existence.",
		"cs50p::fileio",
	),
	(
		"""import shutil
	shutil.copy("a", "b")""",
		"Copy file.",
		"cs50p::fileio",
	),
	(
		"""import tempfile as tf
	tf.NamedTemporaryFile()""",
		"Create temporary file.",
		"cs50p::fileio",
	),
	# --- CS50P::JSON_CSV ---
	(
		"""import csv
	rows = list(csv.reader(open("d.csv")))""",
		"Read CSV rows.",
		"cs50p::json_csv",
	),
	(
		"""with open("d.csv", "w", newline="") as f:
		csv.writer(f).writerow(["a", 1])""",
		"Write CSV row.",
		"cs50p::json_csv",
	),
	(
		"""import json
	obj = json.loads("{\\"a\\":1}")""",
		"Parse JSON string.",
		"cs50p::json_csv",
	),
	(
		"""json.dump({"a": 1}, open("data.json", "w"))""",
		"Write JSON file.",
		"cs50p::json_csv",
	),
	(
		"""import pickle
	pickle.dumps({"a": 1})""",
		"Serialize Python object.",
		"cs50p::json_csv",
	),
	(
		"""pickle.loads(pickle.dumps([1, 2]))""",
		"Round-trip pickle.",
		"cs50p::json_csv",
	),

	# --- CS50P::REGEX ---
	(
		"""import re
	re.search(r"\\d+", "a1b")""",
		"Find digits.",
		"cs50p::regex",
	),
	(
		"""re.findall(r"\\w+", "a b_c")""",
		"Find all word tokens.",
		"cs50p::regex",
	),
	(
		"""re.sub(r"\\s+", " ", "a   b")""",
		"Collapse spaces.",
		"cs50p::regex",
	),
	(
		"""re.split(r",\\s*", "a, b, c")""",
		"Split on comma-space.",
		"cs50p::regex",
	),
	(
		"""pat = re.compile(r"^(\\w+)-(\\d+)$")""",
		"Compile pattern.",
		"cs50p::regex",
	),
	(
		"""m = pat.match("item-42")
	m.groups()""",
		"Capture groups.",
		"cs50p::regex",
	),
	(
		"""re.fullmatch(r"[A-F0-9]+", "DEADBEEF")""",
		"Full string match.",
		"cs50p::regex",
	),
	(
		"""re.subn(r"a", "x", "banana")""",
		"Replace with count.",
		"cs50p::regex",
	),

	# --- CS50P::TESTING ---
	(
		"""def add(a, b):
		\"\"\">>> add(2, 3)
		5
		\"\"\"
		return a + b""",
		"Docstring doctest example.",
		"cs50p::testing",
	),
	(
		"""import doctest
	doctest.testmod()""",
		"Run doctests.",
		"cs50p::testing",
	),
	(
		"""import unittest

	class T(unittest.TestCase):
		def test_add(self):
			self.assertEqual(add(2, 3), 5)""",
		"unittest class definition.",
		"cs50p::testing",
	),
	(
		"""if __name__ == "__main__":
		unittest.main()""",
		"Run unittest main.",
		"cs50p::testing",
	),
	(
		"""def test_even():
		assert 4 % 2 == 0""",
		"pytest-style assertion.",
		"cs50p::testing",
	),
	(
		"""pytest -q""",
		"Run pytest quietly.",
		"cs50p::testing",
	),

	# --- CS50P::MODULES ---
	(
		"""import pkg.util as u""",
		"Import with alias.",
		"cs50p::modules",
	),
	(
		"""from math import pi as PI""",
		"Import symbol alias.",
		"cs50p::modules",
	),
	(
		"""if __name__ == "__main__":
		main()""",
		"Script entry point.",
		"cs50p::modules",
	),
	(
		"""__all__ = ["public_fn"]""",
		"Restrict exports.",
		"cs50p::modules",
	),
	(
		"""import importlib
	importlib.reload(mod)""",
		"Reload module.",
		"cs50p::modules",
	),

	# --- CS50P::ENVIRONMENTS ---
	(
		"""python -m venv .venv""",
		"Create venv.",
		"cs50p::environments",
	),
	(
		"""source .venv/bin/activate""",
		"Activate venv (Unix).",
		"cs50p::environments",
	),
	(
		"""pip install requests""",
		"Install package.",
		"cs50p::environments",
	),
	(
		"""pip freeze > requirements.txt""",
		"Export dependencies.",
		"cs50p::environments",
	),
	(
		"""pip install -r requirements.txt""",
		"Install from requirements file.",
		"cs50p::environments",
	),

	# --- CS50P::ARGPARSE ---
	(
		"""import argparse as ap
	p = ap.ArgumentParser()""",
		"Create parser.",
		"cs50p::argparse",
	),
	(
		"""p.add_argument("--k", type=int, default=1)""",
		"Add argument.",
		"cs50p::argparse",
	),
	(
		"""ns = p.parse_args(["--k", "3"])""",
		"Parse arguments.",
		"cs50p::argparse",
	),
	(
		"""ns.k""",
		"Access parsed value.",
		"cs50p::argparse",
	),

	# --- CS50P::DATACLASSES ---
	(
		"""from dataclasses import dataclass

	@dataclass
	class P:
		x: int
		y: int""",
		"Dataclass model.",
		"cs50p::dataclasses",
	),
	(
		"""P(1, 2)""",
		"Instantiate dataclass.",
		"cs50p::dataclasses",
	),
	(
		"""from typing import Optional, List""",
		"Type hints for functions.",
		"cs50p::dataclasses",
	),
	(
		"""def f(x: int) -> str:
		return str(x)""",
		"Annotated function.",
		"cs50p::dataclasses",
	),
	(
		"""from typing import Iterable

	def g(xs: Iterable[int]):
		pass""",
		"Generic Iterable type.",
		"cs50p::dataclasses",
	),
	(
		"""from typing import TypedDict

	class User(TypedDict):
		name: str
		id: int""",
		"TypedDict definition.",
		"cs50p::dataclasses",
	),
	(
		"""from typing import Protocol

	class Sizable(Protocol):
		def __len__(self) -> int:
			...""",
		"Protocol definition.",
		"cs50p::dataclasses",
	),
	# --- CS50P::OOP_BASICS ---
	(
		"""class Dog:
		def bark(self):
			return "woof" """,
		"Simple class with method.",
		"cs50p::oop_basics",
	),
	(
		"""d = Dog()
	d.bark()""",
		"Call method on instance.",
		"cs50p::oop_basics",
	),
	(
		"""class Dog:
		def __init__(self, name):
			self.name = name""",
		"Constructor method.",
		"cs50p::oop_basics",
	),
	(
		"""class Dog:
		species = "Canis"
		@classmethod
		def kingdom(cls):
			return "Animalia" """,
		"Class variable + method.",
		"cs50p::oop_basics",
	),
	(
		"""class Math:
		@staticmethod
		def add(a, b):
			return a + b""",
		"Static method.",
		"cs50p::oop_basics",
	),
	(
		"""class Dog:
		def __str__(self):
			return f"Dog {self.name}" """,
		"String representation method.",
		"cs50p::oop_basics",
	),
	(
		"""class Cat(Dog):
		def bark(self):
			return "meow" """,
		"Subclass method override.",
		"cs50p::oop_basics",
	),
	(
		"""isinstance(Cat("x"), Dog)""",
		"Instance check under inheritance.",
		"cs50p::oop_basics",
	),
	(
		"""issubclass(Cat, Dog)""",
		"Subclass relationship check.",
		"cs50p::oop_basics",
	),
	(
		"""super(Cat, self).__init__("kit")""",
		"Call parent class constructor.",
		"cs50p::oop_basics",
	),

	# --- CS50P::OOP_ADVANCED ---
	(
		"""class C:
		def __init__(self):
			self._x = 0
		@property
		def x(self):
			return self._x
		@x.setter
		def x(self, v):
			self._x = v""",
		"Property getter/setter.",
		"cs50p::oop_advanced",
	),
	(
		"""class S:
		__slots__ = ("x", "y")
		def __init__(self):
			self.x = 0
			self.y = 0""",
		"Memory optimization with slots.",
		"cs50p::oop_advanced",
	),
	(
		"""class P:
		def __init__(self):
			self.__secret = 42""",
		"Private name mangling.",
		"cs50p::oop_advanced",
	),
	(
		"""class Bag:
		def __init__(self, items):
			self.items = items
		def __len__(self):
			return len(self.items)
		def __getitem__(self, i):
			return self.items[i]""",
		"Sequence protocol methods.",
		"cs50p::oop_advanced",
	),
	(
		"""class R:
		def __enter__(self):
			return self
		def __exit__(self, *a):
			pass

	with R() as r:
		pass""",
		"Context manager protocol.",
		"cs50p::oop_advanced",
	),
	(
		"""class V:
		def __add__(self, o):
			return V(self.x + o.x)""",
		"Operator overloading.",
		"cs50p::oop_advanced",
	),
	(
		"""class V:
		def __repr__(self):
			return f"V({self.x})" """,
		"Debug representation.",
		"cs50p::oop_advanced",
	),
	(
		"""class V:
		def __eq__(self, o):
			return self.x == o.x""",
		"Equality comparison override.",
		"cs50p::oop_advanced",
	),

	# --- CS50P::CONTEXTLIB ---
	(
		"""from contextlib import contextmanager

	@contextmanager
	def sup():
		print("enter")
		yield
		print("exit")""",
		"Contextmanager decorator.",
		"cs50p::contextlib",
	),
	(
		"""with sup():
		pass""",
		"Executes enter/exit of contextmanager.",
		"cs50p::contextlib",
	),

	# --- CS50P::COLLECTIONS ---
	(
		"""from collections import deque
	q = deque([1, 2])
	q.appendleft(0)""",
		"Fast append-left.",
		"cs50p::collections",
	),
	(
		"""q.popleft()""",
		"Pop from left side.",
		"cs50p::collections",
	),
	(
		"""from collections import OrderedDict
	list(OrderedDict.fromkeys("banana"))""",
		"Stable deduplication.",
		"cs50p::collections",
	),
	(
		"""from collections import ChainMap
	ChainMap({"a": 1}, {"a": 2})["a"]""",
		"First mapping wins.",
		"cs50p::collections",
	),
	(
		"""import itertools as it
	dict(it.zip_longest("ab", "xyz", fillvalue="-"))""",
		"Zip uneven iterables.",
		"cs50p::collections",
	),

	# --- CS50P::TIME ---
	(
		"""import time
	time.perf_counter()""",
		"High-resolution timer.",
		"cs50p::time",
	),
	(
		"""from time import sleep
	sleep(0.1)""",
		"Pause execution.",
		"cs50p::time",
	),
	(
		"""from datetime import datetime as dt, timedelta
	(dt.now() + timedelta(days=1)).date()""",
		"Date arithmetic.",
		"cs50p::time",
	),
	(
		"""from datetime import datetime as dt
	dt.fromisoformat("2025-10-30")""",
		"Parse ISO date.",
		"cs50p::time",
	),

	# --- CS50P::PERFORMANCE ---
	(
		"""import time
	t = time.perf_counter()
	work()
	print(time.perf_counter() - t)""",
		"Manual timing.",
		"cs50p::performance",
	),
	(
		"""from functools import lru_cache

	@lru_cache(None)
	def slow(x):
		...""",
		"Memoize function.",
		"cs50p::performance",
	),
	(
		"""sum(x * x for x in range(10 ** 6))""",
		"Generator saves memory.",
		"cs50p::performance",
	),
	(
		"""list(map(str, range(10)))""",
		"Lazy map usage.",
		"cs50p::performance",
	),

	# --- CS50P::ASYNCIO ---
	(
		"""import asyncio

	async def work():
		return 1

	asyncio.run(work())""",
		"Run coroutine.",
		"cs50p::asyncio",
	),
	(
		"""import asyncio

	async def fetch():
		await asyncio.sleep(0.1)
		return "ok" """,
		"Await async function.",
		"cs50p::asyncio",
	),
	(
		"""import asyncio

	async def main():
		a = fetch()
		b = fetch()
		r = await asyncio.gather(a, b)
		return r""",
		"Concurrent tasks.",
		"cs50p::asyncio",
	),

	# --- CS50P::THREADING ---
	(
		"""import threading as th
	lock = th.Lock()""",
		"Threading lock object.",
		"cs50p::threading",
	),
	(
		"""import threading as th
	t = th.Thread(target=lambda: None)
	t.start()
	t.join()""",
		"Start/join thread.",
		"cs50p::threading",
	),
	(
		"""from concurrent.futures import ThreadPoolExecutor as TPE

	with TPE() as ex:
		list(ex.map(int, ["1", "2"]))""",
		"ThreadPoolExecutor map.",
		"cs50p::threading",
	),
	(
		"""from concurrent.futures import ProcessPoolExecutor as PPE

	with PPE() as ex:
		list(ex.map(pow, [2, 2], [10, 11]))""",
		"ProcessPoolExecutor map.",
		"cs50p::threading",
	),

	# --- CS50P::LOGGING ---
	(
		"""import logging as log
	log.basicConfig(level=log.INFO)""",
		"Configure logger.",
		"cs50p::logging",
	),
	(
		"""import logging as log
	log.info("message")""",
		"Log info message.",
		"cs50p::logging",
	),
	(
		"""import logging as log
	log.exception("fail")""",
		"Log with traceback.",
		"cs50p::logging",
	),
	# --- CS50P::SQLITE ---
	(
		"""import sqlite3 as sq
	con = sq.connect(":memory:")
	cur = con.cursor()""",
		"Connect to SQLite memory DB.",
		"cs50p::sqlite",
	),
	(
		"""cur.execute("CREATE TABLE t(x)")""",
		"Create table.",
		"cs50p::sqlite",
	),
	(
		"""cur.executemany("INSERT INTO t(x) VALUES(?)", [(1,), (2,)])""",
		"Bulk insert.",
		"cs50p::sqlite",
	),
	(
		"""list(cur.execute("SELECT * FROM t"))""",
		"Query rows.",
		"cs50p::sqlite",
	),
	(
		"""con.commit()
	con.close()""",
		"Commit and close DB.",
		"cs50p::sqlite",
	),

	# --- CS50P::NETWORK ---
	(
		"""import urllib.request as u
	u.urlopen("https://example.com").read()[:15]""",
		"HTTP GET via urllib.",
		"cs50p::network",
	),
	(
		"""# requests.get("https://api")""",
		"Third-party requests pattern.",
		"cs50p::network",
	),

	# --- CS50P::JSONL ---
	(
		"""import json
	with open("items.jsonl") as f:
		rows = [json.loads(line) for line in f]""",
		"Load JSONL file.",
		"cs50p::jsonl",
	),
	(
		"""with open("out.jsonl", "w") as f:
		f.write(json.dumps({"a": 1}) + "\\n")""",
		"Write JSONL line.",
		"cs50p::jsonl",
	),

	# --- CS50P::PATHOPS ---
	(
		"""from tempfile import TemporaryDirectory as TD
	from pathlib import Path

	with TD() as d:
		p = Path(d) / "x.txt"
		p.write_text("ok")""",
		"Use temporary directory.",
		"cs50p::pathops",
	),
	(
		"""from pathlib import Path
	Path("dir").mkdir(parents=True, exist_ok=True)""",
		"Create directories safely.",
		"cs50p::pathops",
	),

	# --- CS50P::ENVVARS ---
	(
		"""import os
	os.environ.get("HOME")""",
		"Read environment variable.",
		"cs50p::envvars",
	),
	(
		"""import os
	os.environ["MODE"] = "dev" """,
		"Set environment variable.",
		"cs50p::envvars",
	),

	# --- CS50P::ERRORHANDLING ---
	(
		"""from contextlib import suppress
	from json import loads

	with suppress(ValueError):
		loads("bad")""",
		"Suppress specific errors.",
		"cs50p::errorhandling",
	),
	(
		"""import time
	for i in range(3):
		try:
			risky()
			break
		except OSError:
			time.sleep(0.1)""",
		"Retry loop with backoff.",
		"cs50p::errorhandling",
	),

	# --- CS50P::COMPREHENSIONS ---
	(
		"""[(i, j) for i in range(3) for j in range(i)]""",
		"Nested list comprehension.",
		"cs50p::comprehensions",
	),
	(
		"""{k: v for k, v in d.items() if v % 2 == 0}""",
		"Filtered dict comprehension.",
		"cs50p::comprehensions",
	),
	(
		"""[(i, i * i) for i in range(5) if i % 2 == 0]""",
		"Select and transform.",
		"cs50p::comprehensions",
	),
	(
		"""{c for c in "aabbcc"}""",
		"Unique chars set.",
		"cs50p::comprehensions",
	),
	(
		"""def chunks(seq, n):
		for i in range(0, len(seq), n):
			yield seq[i:i+n]""",
		"Yield chunked sequence.",
		"cs50p::comprehensions",
	),
	(
		"""from collections import deque

	def tail(it, n):
		return deque(it, maxlen=n)""",
		"Keep last n items.",
		"cs50p::comprehensions",
	),

	# --- CS50P::DEBUGGING ---
	(
		"""import pprint as pp
	pp.pprint(obj)""",
		"Pretty-print structures.",
		"cs50p::debugging",
	),
	(
		"""import inspect
	inspect.signature(len)""",
		"Inspect function signature.",
		"cs50p::debugging",
	),
	(
		"""vars(obj)""",
		"Get object attributes dict.",
		"cs50p::debugging",
	),
	(
		"""dir(obj)""",
		"List available attributes.",
		"cs50p::debugging",
	),
	(
		"""help(str)""",
		"Built-in help system.",
		"cs50p::debugging",
	),

	# --- CS50P::PATTERNS ---
	(
		"""from enum import Enum

	class State(Enum):
		READY = 1
		RUN = 2""",
		"Enum pattern.",
		"cs50p::patterns",
	),
	(
		"""match obj:
		case {"type": "user", "id": uid}:
			use(uid)""",
		"Dict structural pattern match.",
		"cs50p::patterns",
	),
	(
		"""match seq:
		case [x, y, *rest]:
			...""",
		"List pattern match.",
		"cs50p::patterns",
	),

	# --- CS50P::IMMUTABLE ---
	(
		"""fs = frozenset({1, 2, 3})""",
		"Immutable set.",
		"cs50p::immutable",
	),
	(
		"""from dataclasses import dataclass

	@dataclass(frozen=True)
	class Point:
		x: int
		y: int""",
		"Frozen dataclass.",
		"cs50p::immutable",
	),

	# --- CS50P::ALGORITHMS ---
	(
		"""def binary_search(a, x):
		lo, hi = 0, len(a) - 1
		while lo <= hi:
			mid = (lo + hi) // 2
			if a[mid] == x:
				return mid
			if a[mid] < x:
				lo = mid + 1
			else:
				hi = mid - 1
		return -1""",
		"Binary search algorithm.",
		"cs50p::algorithms",
	),
	(
		"""def two_sum(nums, target):
		seen = {}
		for i, n in enumerate(nums):
			if target - n in seen:
				return (seen[target - n], i)
			seen[n] = i""",
		"Hash map two-sum.",
		"cs50p::algorithms",
	),
	(
		"""def is_pal(s):
		return s == s[::-1]""",
		"Palindrome check.",
		"cs50p::algorithms",
	),

	# --- CS50P::UTILITIES ---
	(
		"""d = {"a": 1, "b": 2}
	{**d, "b": 3}""",
		"Dict merge.",
		"cs50p::utilities",
	),
	(
		"""defaults | overrides""",
		"Dict union (3.9+).",
		"cs50p::utilities",
	),
	(
		"""[x for x in nums if x is not None]""",
		"Filter None values.",
		"cs50p::utilities",
	),
	(
		"""any(x > 10 for x in data)""",
		"Any condition true.",
		"cs50p::utilities",
	),
	(
		"""all(x >= 0 for x in data)""",
		"All conditions true.",
		"cs50p::utilities",
	),
	(
		"""next((x for x in data if x > 10), None)""",
		"Find first matching element.",
		"cs50p::utilities",
	),
	(
		"""print((1, 2) * 2)""",
		"Tuple repetition.",
		"cs50p::utilities",
	),
	(
		"""print([[]] * 3)""",
		"Mutable list reference trap.",
		"cs50p::utilities",
	),
	(
		"""sum([]) == 0""",
		"Sum of empty list is 0.",
		"cs50p::utilities",
	),
	(
		"""min([], default=None)""",
		"Safe min with default.",
		"cs50p::utilities",
	),
]
