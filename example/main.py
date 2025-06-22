if __name__ == "__main__":
    from host import main

    main()
else:
    import os
    import site
    import sys

    if os.name == "nt":
        venv = os.environ.get("VIRTUAL_ENV", "")
        if venv != "":
            sys.path.insert(0, os.path.join(venv, "Lib", "site-packages"))
            site.addsitedir(os.path.join(venv, "Lib", "site-packages"))
