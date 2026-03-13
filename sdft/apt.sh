#!/bin/bash

rm /etc/apt/sources.list

cp sources.list /etc/apt/sources.list

apt update

apt install unzip

apt install tmux