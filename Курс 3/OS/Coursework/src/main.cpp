#include "PcapLiveDeviceList.h"
#include "SystemUtils.h"
#include "stdlib.h"
#include <IPv4Layer.h>
#include <IPv6Layer.h>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <ostream>

struct PacketStats {
    int ethPacketCount = 0;
    int ipv4PacketCount = 0;
    int ipv6PacketCount = 0;
    int tcpPacketCount = 0;
    int udpPacketCount = 0;
    int dnsPacketCount = 0;
    int httpPacketCount = 0;
    int sslPacketCount = 0;

    void clear() { ethPacketCount = ipv4PacketCount = ipv6PacketCount = tcpPacketCount = udpPacketCount = dnsPacketCount = httpPacketCount = sslPacketCount = 0; }

    PacketStats() = default;

    void consumePacket(pcpp::Packet& packet)
    {
        if (packet.isPacketOfType(pcpp::Ethernet))
            ethPacketCount++;
        if (packet.isPacketOfType(pcpp::IPv4))
            ipv4PacketCount++;
        if (packet.isPacketOfType(pcpp::IPv6))
            ipv6PacketCount++;
        if (packet.isPacketOfType(pcpp::TCP))
            tcpPacketCount++;
        if (packet.isPacketOfType(pcpp::UDP))
            udpPacketCount++;
        if (packet.isPacketOfType(pcpp::DNS))
            dnsPacketCount++;
        if (packet.isPacketOfType(pcpp::HTTP))
            httpPacketCount++;
        if (packet.isPacketOfType(pcpp::SSL))
            sslPacketCount++;
    }

    void printToConsole()
    {
        std::cout
            << "Ethernet packet count: " << ethPacketCount << std::endl
            << "IPv4 packet count:     " << ipv4PacketCount << std::endl
            << "IPv6 packet count:     " << ipv6PacketCount << std::endl
            << "TCP packet count:      " << tcpPacketCount << std::endl
            << "UDP packet count:      " << udpPacketCount << std::endl
            << "DNS packet count:      " << dnsPacketCount << std::endl
            << "HTTP packet count:     " << httpPacketCount << std::endl
            << "SSL packet count:      " << sslPacketCount << std::endl;
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

    std::system("clear");

    // Main device
    auto selectedDevice = devices[number_device - 1];

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

    std::cout << std::endl
              << "Starting capture with packet vector..." << std::endl;

    // create an empty packet vector object
    pcpp::RawPacketVector packetVec;

    // start capturing packets. All packets will be added to the packet vector
    selectedDevice->startCapture(packetVec);

    // sleep for 10 seconds in main thread, in the meantime packets are captured in the async thread
    pcpp::multiPlatformSleep(10);

    // stop capturing packets
    selectedDevice->stopCapture();

    std::cout << std::endl;

    std::cout << "IP Address packet : " << std::endl;

    for (const auto& packet : packetVec) {
        pcpp::Packet parsedPacket(packet);
        stats.consumePacket(parsedPacket);

        if (parsedPacket.isPacketOfType(pcpp::IPv4)) {
            auto ipv4Header = parsedPacket.getLayerOfType<pcpp::IPv4Layer>();
            if (ipv4Header != nullptr) {
                std::cout << "IPv4 Packet: " << ipv4Header->getSrcIPv4Address() << " -> " << ipv4Header->getDstIPAddress() << std::endl;
            }
        } else if (parsedPacket.isPacketOfType(pcpp::IPv6)) {
            auto ipv6Header = parsedPacket.getLayerOfType<pcpp::IPv6Layer>();
            if (ipv6Header != nullptr) {
                std::cout << "IPv6 Packet: " << ipv6Header->getSrcIPAddress() << " -> " << ipv6Header->getDstIPAddress() << std::endl;
            }
        }
    }

    std::cout << std::endl;

    // Print results
    std::cout << "Results:" << std::endl;
    stats.printToConsole();
}
