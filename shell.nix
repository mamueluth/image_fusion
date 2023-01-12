with import <nixpkgs> {};
with python310Packages;

let 
 opencv4_gtk = pkgs.python310Packages.opencv4.override { enableGtk3 = true; enableCuda = true; };
 matplotlib_gtk = pkgs.python310Packages.matplotlib.override {enableTk = true; };
in
stdenv.mkDerivation {
  name = "image_fusion";
  src = null;

  # install packages which need to be installed through nix like opencv4
  buildInputs = [
    virtualenv
    setuptools
    venvShellHook
    opencv4_gtk
    matplotlib_gtk
    ];

  shellHook = ''
    VENV=.venv
    REQUIREMENTS=requirements.txt
 
    if [[ ! -d $VENV ]] 
    then
      virtualenv --no-setuptools $VENV
    fi
 
    # setup virtual env and update packages
    source ./$VENV/bin/activate
    python -m pip install --upgrade pip
 
    # install requirements for venv
    if [[ -f $REQUIREMENTS ]]
    then
      python -m pip install -r $REQUIREMENTS
    fi
    export PATH=$PWD/venv/bin:$PATH
    export PYTHONPATH=venv/lib/python3.10/site-packages/:$PYTHONPATH
  '';

  # expose all installed packages into virtual env
  postShellHook = ''
    ln -sf PYTHONPATH/* ${virtualenv}/lib/python3.10/site-packages
  '';
}
