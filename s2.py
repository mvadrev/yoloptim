import asyncio
import wave
import audioop

from aiortc.rtp import RtpPacket
from aiortc.rtcrtpsender import RTCRtpSender
from aiortc.rtcrtpparameters import RTCRtpCodecParameters
from aiortc.rtcrtpsender import RTCRtpSendParameters

import socket

DST_IP = "192.168.1.100"
DST_PORT = 5004
WAV_FILE = "audio_mono_8000hz.wav"

# PCMU payload type standard is 0
PCMU_PAYLOAD_TYPE = 0

def create_rtp_packets(wav_path):
    wf = wave.open(wav_path, 'rb')

    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 8000:
        raise ValueError("WAV must be mono, 16-bit, 8kHz")

    frames_per_packet = 160  # 20ms at 8000Hz
    seq = 0
    timestamp = 0
    ssrc = 12345

    while True:
        pcm = wf.readframes(frames_per_packet)
        if len(pcm) < frames_per_packet * 2:
            break

        # Encode to PCMU (μ-law)
        ulaw = audioop.lin2ulaw(pcm, 2)

        # Create RTP packet
        packet = RtpPacket(
            payload_type=PCMU_PAYLOAD_TYPE,
            sequence_number=seq,
            timestamp=timestamp,
            ssrc=ssrc,
            payload=ulaw
        )

        yield packet

        seq = (seq + 1) % 65536
        timestamp += frames_per_packet

    wf.close()

async def send_rtp():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    for packet in create_rtp_packets(WAV_FILE):
        sock.sendto(packet.serialize(), (DST_IP, DST_PORT))
        await asyncio.sleep(0.02)  # 20ms

    sock.close()

if __name__ == "__main__":
    asyncio.run(send_rtp())
