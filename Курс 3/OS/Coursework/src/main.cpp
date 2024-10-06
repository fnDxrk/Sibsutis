#include "PcapLiveDeviceList.h"
#include "SystemUtils.h"
#include "stdlib.h"
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <ostream>

int main(int argc, char* argv[])
{
    auto devices = pcpp::PcapLiveDeviceList::getInstance().getPcapLiveDevicesList();

    if (devices.empty()) {
        std::cerr << "No devices found." << std::endl;
        return 1;
    }

    // List devices
    std::cout << "List devices :" << std::endl;
    std::size_t index = 0;
    for (const auto& dev : devices) {
        std::cout << index + 1 << ". Name : " << dev->getName() << std::endl;
        index++;
    }
    std::cout << std::endl;

    std::cout << "Enter the number of the selected device : ";
    int number_device;
    std::cin >> number_device;

    if (std::cin.fail() || number_device < 1 || number_device > devices.size()) {
        std::cerr << "Invalid selection. Please enter a number between 1 and "
                  << devices.size() << ".\n";
        return 1;
    }

    std::system("clear");

    // Main device
    auto selectedDevice = devices[number_device - 1];
}
