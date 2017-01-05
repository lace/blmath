skeleton-py
===========

A project starter for Body Labs Python projects.

To use this skeleton:

1. Clone the skeleton:

        git clone git@github.com:bodylabs/template-py.git my-library

2. Re-initialize the git repository:

        cd my-library && rm -rf .git && git init

3. Publish the license, or delete it, depending whether or not the project is
   open source:

        mv RENAME-TO-LICENSE-IF-PUBLIC-ELSE-DELETE LICENSE
        rm RENAME-TO-LICENSE-IF-PUBLIC-ELSE-DELETE

4. Commit the changed files.

        git add . && git commit -m "Project skeleton"

5. Make updates as needed.


- - - - - - - - - - - - -

example
=======

Short description of the module.


Features
--------

- This
- That
- The other


Examples
--------

```py
from example.hello import hello
hello()
```

```sh
hello
```


Development
-----------

```sh
pip install -r requirements_dev.txt
rake test
rake lint
```


Contribute
----------

- Issue Tracker: github.com/bodylabs/example/issues
- Source Code: github.com/bodylabs/example

Pull requests welcome!


Support
-------

If you are having issues, please let us know.


License
-------

The project is licensed under the two-clause BSD license.
