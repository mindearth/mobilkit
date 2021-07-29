# Compile documentation

Install `pandoc` in your system (e.g., `apt-get install pandoc` on ubuntu) and run

```
make html
```

from within the `docs/` folder. This will create the `_build/html` folder to be served on the server hosting the documentation.
You can change the target of your build to pdf or other with `make pdf`.

# Add notebooks/pages to the documentation

Add the desired noebooks in the `docs/examples` folder and then add them to the `..toc` directive in the `index.rst`. Then see point above to rebuild the documentation.
