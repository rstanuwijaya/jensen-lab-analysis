# %%

from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
    ipython.run_line_magic("matplotlib", "inline")

from drivers import meta

def main():
    meta.main()
    # meta.test_main()


if __name__ == '__main__':
    main()
# %%
