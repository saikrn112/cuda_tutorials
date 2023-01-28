This dockerfile is for tensorflow image, by default it tries to attach my workspace directory. You can attach your own workspace by passing appropriate flags in the launch script (more on this below)

To run:
* install docker from [here](https://docs.docker.com/engine/install/)
* also follow [post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/) for linux if you dont want to run docker in `sudo` 
* install docker python package to make life easy for controlling docker
```
pip install docker
```
* build the image
```
./build.sh
```
* run the python script to start and connect to the instance
```
python ./start_docker_instance.py 
```
* for more options and defaults 
```
python ./start_docker_instance.py --help
```

Following are some of the help options 
* for custom container name `--name <name_you_want>` 
* for interactive shell `--it` 
* for attaching your own volume `--workspace <absolute_path_to_your_ws>` 
* to get display out of docker give `--display` 
* example: to run interactive shell with display, run the following
```
#make sure existing docker image is already not spinned
python ./start_docker_instance.py --it --display
```
