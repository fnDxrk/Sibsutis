#include "PcapLiveDeviceList.h"
#include "SystemUtils.h"
#include "stdlib.h"
#include <IPv4Layer.h>
#include <IPv6Layer.h>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <ostream>

struct PacketStats {
    std::array<int, 8> packetCounts = { 0 }; // 0: Ethernet, 1: IPv4, ..., 7: SSL

    void clear() { packetCounts.fill(0); }

    void consumePacket(pcpp::Packet& packet)
    {
        if (packet.isPacketOfType(pcpp::Ethernet))
            packetCounts[0]++;
        if (packet.isPacketOfType(pcpp::IPv4))
            packetCounts[1]++;
        if (packet.isPacketOfType(pcpp::IPv6))
            packetCounts[2]++;
        if (packet.isPacketOfType(pcpp::TCP))
            packetCounts[3]++;
        if (packet.isPacketOfType(pcpp::UDP))
            packetCounts[4]++;
        if (packet.isPacketOfType(pcpp::DNS))
            packetCounts[5]++;
        if (packet.isPacketOfType(pcpp::HTTP))
            packetCounts[6]++;
        if (packet.isPacketOfType(pcpp::SSL))
            packetCounts[7]++;
    }

    void printToConsole()
    {
        const char* labels[] = {
            "Ethernet", "IPv4", "IPv6", "TCP", "UDP", "DNS", "HTTP", "SSL"
        };
        for (size_t i = 0; i < packetCounts.size(); ++i) {
            std::cout << labels[i] << " packet count: " << packetCounts[i] << std::endl;
        }
    }
};

int main(int argc, char* argv[])
{
    // Create the stats object
    PacketStats stats;

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

    std::cout << "\033[2J\033[1;1H";

    // Main device
    auto selectedDevice
        = devices[number_device - 1];

    // Open the device before start capturing/sending packets
    if (!selectedDevice->open()) {
        std::cerr << "Cannot open device" << std::endl;
        return 1;
    }

    std::cout
        << "Interface info:" << std::endl
        << "   Interface name:        " << selectedDevice->getName() << std::endl // get interface name
        << "   Interface description: " << selectedDevice->getDesc() << std::endl // get interface description
        << "   MAC address:           " << selectedDevice->getMacAddress() << std::endl // get interface MAC address
        << "   Default gateway:       " << selectedDevice->getDefaultGateway() << std::endl // get default gateway
        << "   Interface MTU:         " << selectedDevice->getMtu() << std::endl; // get interface MTU

    std::cout << "\nStarting capture with packet vector..." << std::endl;

    // create an empty packet vector object
    pcpp::RawPacketVector packetVec;

    // start capturing packets. All packets will be added to the packet vector
    selectedDevice->startCapture(packetVec);

    // sleep for 10 seconds in main thread, in the meantime packets are captured in the async thread
    pcpp::multiPlatformSleep(10);

    // stop capturing packets
    selectedDevice->stopCapture();

    std::cout << "\nIP Address packet : " << std::endl;

    for (const auto& packet : packetVec) {
        pcpp::Packet parsedPacket(packet);
        stats.consumePacket(parsedPacket);

        if (auto ipv4Header = parsedPacket.getLayerOfType<pcpp::IPv4Layer>(); ipv4Header) {
            std::cout << "IPv4 Packet: " << ipv4Header->getSrcIPv4Address() << " -> " << ipv4Header->getDstIPAddress() << std::endl;
        } else if (auto ipv6Header = parsedPacket.getLayerOfType<pcpp::IPv6Layer>(); ipv6Header) {
            std::cout << "IPv6 Packet: " << ipv6Header->getSrcIPAddress() << " -> " << ipv6Header->getDstIPAddress() << std::endl;
        }
    }

    // Print results
    std::cout << "\nResults:" << std::endl;
    stats.printToConsole();
}
