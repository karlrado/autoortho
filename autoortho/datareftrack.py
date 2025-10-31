#!/usr/bin/env python3

"""
Description: Collect datarefs from X-Plane.
"""

import binascii
import time
import struct
import socket
import threading
from aoconfig import CFG
import logging

log = logging.getLogger(__name__)
UDP_IP = "127.0.0.1"


class DatarefTracker(object):
    lat = -1.0
    lon = -1.0
    alt = -1.0
    hdg = -1.0
    spd = -1.0
    t = None
    udp_timeout = 5
    connected = False
    data_valid = False

    # List of datarefs to request.
    # fmt:off
    datarefs = [
        # ( dataref, unit, description, num decimals to display in formatted output )
        ( "sim/flightmodel/position/latitude",         "°N",
          "The latitude of the aircraft",              6 ),
        ( "sim/flightmodel/position/longitude",        "°E",
          "The longitude of the aircraft",             6 ),
        ( "sim/flightmodel/position/y_agl",            "m",
            "AGL",                                     0 ),
        ( "sim/flightmodel/position/mag_psi",          "°",
          "The real magnetic heading of the aircraft", 0 ),
        ( "sim/flightmodel/position/groundspeed",      "m/s",
          "The ground speed of the aircraft",          0 ),
    ]
    # fmt:on

    def __init__(self):
        self.sock = socket.socket(
            socket.AF_INET,  # Internet
            socket.SOCK_DGRAM,  # UDP
        )

        self.sock.settimeout(self.udp_timeout)
        self.connected = False
        self.running = False
        log.info("Dataref Tracker instance created")

    def start(self):
        if self.running:
            return
        log.info("Starting UDP listening thread")
        self.running = True
        self.t = threading.Thread(target=self._dataref_tracker_udp_listen)
        self.t.start()

    def stop(self):
        log.info("Dataref Tracker shutdown requested.")
        # Unsubscribe
        # It is possible that X-Plane is already shut down, so ignore any errors
        # since we don't care at this point anyway.
        # We can't really check if the socket is still open first, since XP could
        # shutdown between the check and the unsub request.
        try:
            self.RequestDataRefs(self.sock, CFG.flightdata.xplane_udp_port, 0)
        except (Exception, OSError):
            pass
        self.running = False
        if self.t:
            self.t.join()
        self.sock.close()
        log.info("Thread terminating.")

    # We're really only interested in whether or not X-Plane is able to send us
    # datarefs, which it only does when a flight is active.  Note that a flight is
    # not considered active when X-Plane is displaying the
    # "Reading new scenery files" splash screen before a flight.
    #
    # - For the feature where we increase maxwait during the initial scenery load,
    #   we just need to know if the flight is active (could be flying).  When it is
    #   not active (e.g., at a splash screen before the flight), we know we are not
    #   flying and can increase maxwait without the risk of a stutter.
    #
    # - For the predictive preload feature, we need to know if the flight is active
    #   so we can get good datarefs to predict the flight path.
    #
    # We don't really need to know if X-Plane is simply up and running.  So there
    # really isn't any need to use the "beacon" feature to "find" a running instance
    # of X-Plane.  If we ever have a need to do so, here are some resources:
    # https://github.com/charlylima/XPlaneUDP/blob/master/XPlaneUdp.py
    # https://gitlab.bliesener.com/jbliesener/PiDisplay/-/blob/master/XPlaneUDP.py
    #
    # It therefore seems like the best way to detect when we can get datarefs is to
    # just periodically subscribe for datarefs and see if a response comes back in a
    # reasonable time.  If it does not, clear/reset the socket and try again after
    # a short wait.
    def _dataref_tracker_udp_listen(self):
        log.info("UDP listening thread started and listening!")
        # Subscribe to dataref updates at the default rate
        self.RequestDataRefs(self.sock, CFG.flightdata.xplane_udp_port)
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
            except Exception:
                # All exceptions including ConnectionResetError and socket.timeout
                # are handled in the same way.
                # This can happen when X-Plane is not running yet or has stopped.
                # For now, reset the socket, wait a bit (quietly), and retry.
                if self.connected:
                    log.info("Flight has stopped.")
                self.connected = False
                self.data_valid = False
                self.sock.close()
                self.sock = socket.socket(
                    socket.AF_INET,  # Internet
                    socket.SOCK_DGRAM,  # UDP
                )
                self.sock.settimeout(self.udp_timeout)
                time.sleep(5)
                self.RequestDataRefs(self.sock, CFG.flightdata.xplane_udp_port)
                continue

            # Socket read was successful
            if not self.connected:
                log.info("Flight is starting.")
                self.connected = True

            values = self.DecodePacket(data)
            if len(values) == 5:
                self.lat = values[0][0]
                self.lon = values[1][0]
                self.alt = values[2][0]
                self.hdg = values[3][0]
                self.spd = values[4][0]
                self.data_valid = True
            else:
                self.data_valid = False

            # log.info(
            #     f"Lat: {self.lat:.6f}, Lon: {self.lon:.6f}, Alt: {self.alt:.0f}, "
            #     f"Hdg: {self.hdg:.0f}, Spd: {self.spd:.0f}"
            # )

        # No longer in running state
        log.info("UDP listen thread exiting...")
        # Let the thread exit

    def RequestDataRefs(self, sock, UDP_PORT=49000, REQ_FREQ=1):
        for idx, dataref in enumerate(self.datarefs):
            # Send one RREF Command for every dataref in the list.
            # Give them an index number and a frequency in Hz.
            # To disable sending you send frequency 0.
            cmd = b"RREF\x00"
            freq = int(REQ_FREQ)
            string = self.datarefs[idx][0].encode()
            message = struct.pack("<5sii400s", cmd, freq, idx, string)
            assert len(message) == 413
            sock.sendto(message, (UDP_IP, int(UDP_PORT)))

    def DecodePacket(self, data):
        retvalues = {}
        header = data[0:4]
        if header != b"RREF":
            log.info(f"Unknown packet: {binascii.hexlify(data)}")
        else:
            # We get 8 bytes for every dataref sent:
            #    An integer for idx and the float value.
            values = data[5:]
            lenvalue = 8
            numvalues = int(len(values) / lenvalue)
            idx = 0
            value = 0
            for i in range(0, numvalues):
                singledata = data[(5 + lenvalue * i) : (5 + lenvalue * (i + 1))]
                (idx, value) = struct.unpack("<if", singledata)
                retvalues[idx] = (value, self.datarefs[idx][1], self.datarefs[idx][0])
        return retvalues


dt = DatarefTracker()
