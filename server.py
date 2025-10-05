from pyVoIP import RTPPacket, RTPProtocol
import wave
import socket
import time

# Configuration
DST_IP = "192.168.1.100"   # Receiver IP
DST_PORT = 5004            # Receiver port
WAV_FILE = "test.wav"  # Input WAV file (mono, 8kHz)

def ulaw_encode(sample):
    """Convert 16-bit PCM to 8-bit μ-law."""
    # pyVoIP has built-in ulaw encode, but here is a simple function or you can use library func.
    # Alternatively, install 'audioop' and use audioop.lin2ulaw
    import audioop
    return audioop.lin2ulaw(sample, 2)  # 2 bytes per sample (16-bit PCM)

def main():
    # Open WAV file
    wf = wave.open(WAV_FILE, 'rb')
    if wf.getnchannels() != 1 or wf.getframerate() != 8000 or wf.getsampwidth() != 2:
        print("WAV file must be mono, 16-bit, 8kHz")
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    seq = 0
    timestamp = 0
    ssrc = 12345
    payload_type = 0  # PCMU payload type

    frames_per_packet = 160  # 20ms at 8kHz

    while True:
        raw_data = wf.readframes(frames_per_packet)
        if not raw_data:
            break

        # Encode PCM16 to ulaw8
        ulaw_data = ulaw_encode(raw_data)

        # Create RTP packet
        rtp_packet = RTPPacket(payload_type=payload_type,
                               sequence=seq,
                               timestamp=timestamp,
                               ssrc=ssrc,
                               payload=ulaw_data)

        # Send RTP packet
        sock.sendto(rtp_packet.get_packet(), (DST_IP, DST_PORT))

        seq = (seq + 1) % 65536
        timestamp += frames_per_packet  # timestamp increments by samples count

        time.sleep(0.02)  # 20 ms per packet

    wf.close()
    sock.close()

if __name__ == "__main__":
    main()
