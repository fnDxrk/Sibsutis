#!/bin/bash

echo -e "OS          $(lsb_release -sd | tr -d '"')"
echo -e "Kernel      $(uname -srm)\n"

echo -e "CPU         $(lscpu | grep "Имя модели" | cut -d ':' -f2 | xargs)"
echo -e "CPU MHz     $(cat /proc/cpuinfo | grep "Hz" | cut -d ':' -f2 | head -n 1 | xargs)"
echo -e "CPU Core    $(lscpu | grep "Ядер" | cut -d ':' -f2 | head -n 1 | xargs)"
echo -e "CPU Cache   $(cat /proc/cpuinfo | grep "cache" | cut -d ':' -f2 | head -n 1 | xargs)\n"

echo -e "Mem free    $(free --mega | grep Mem | xargs | cut -d " " -f4) MB"
echo -e "Mem total   $(free --mega | grep Mem | xargs | cut -d " " -f2) MB"
echo -e "Mem used    $(free --mega | grep Mem | xargs | cut -d " " -f3) MB\n"

echo -e "User        $(whoami)"
echo -e "IP          $(ip -br a show | grep UP | xargs | cut -d " " -f3)"
echo -e "MAC address $(ip a | grep ether | tail -n 1 | xargs | cut -d " " -f3)\n"

echo -e "$(df -h | grep ^/dev/ | awk '{printf "%-10s %-10s %s/%s\n", $6, $2, $3, $4}')"

