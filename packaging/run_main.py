import os
import sys
import streamlit.web.bootstrap

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

if __name__ == "__main__":
    # Ensure we are in the right directory
    os.chdir(resource_path("."))

    flag_options = {
        "server.port": 8501,
        "global.developmentMode": False,
    }

    streamlit.web.bootstrap.load_config_options(flag_options=flag_options)
    flag_options["_is_running_with_streamlit"] = True
    
    # Path to the main app script relative to the bundle root
    main_script = resource_path("src/app.py")
    
    streamlit.web.bootstrap.run(
        main_script,
        False,
        ['run'],
        flag_options,
    )
