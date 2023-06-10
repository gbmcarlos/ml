```
sudo apt-get update
sudo apt-get -y upgrade
curl -sL https://deb.nodesource.com/setup_18.x -o /tmp/nodesource_setup.sh
sudo bash /tmp/nodesource_setup.sh
sudo apt-get install -y nodejs
sudo apt-get install -y libgdal-dev
gdal-config --version
pip install --upgrade jupyterlab-git
sudo systemctl restart lambda-jupyter.service
```