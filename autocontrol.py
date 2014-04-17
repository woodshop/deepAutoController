
"""
    Use a Korg nonoKONTROL2 to control the weights of a n autoencoder. Given
    a learned model. The Korg attaches to the input of the center (topmost
    encoding layer). The faders of the Korg may be used to multiply the
    incoming weights. Use the faders as a multiplier, use the knobs to change
    the range of the faders. Use the solo buttons to reverse the sign of the
    multiplier. An output screen shows in real time the scale and multipler
    values.

    The play  button plays the original audio file (reconstructed from the
    inverse CQFT. The record button plays the inverse resynthesized CQFT
    as output by the autoencoder. The stop button quits the application.
"""
import argparse
import pygame
import pygame.midi
import theano
from pylearn2.utils import sharedX
import numpy as np
from matplotlib import pyplot
import os
import sys
from pylearn2.utils import sharedX
import wave
import multiprocessing
import pyaudio
import struct
import cPickle

NEURONS_PER_BANK = 8
MID_BUF = 1024
#CHUNK = 2048
#OLAP = 1536
#NBUFS = CHUNK / (CHUNK - OLAP) # 4

os.environ['SDL_VIDEO_WINDOW_POS'] = "0,400"
class Autocontrol(object):
    def __init__(self, model_file, wav_file, Q, plot=False, linear=False):
    	self.model_file = model_file
    	self.wav_file = wav_file
        self.queue = Q
        while True:
            cmd,msg = self.queue.get(True) if not self.queue.empty() else ['',None]
            if cmd == 'nneurons':
                self.nneurons = msg
                break
        #self.plotting = plot
        self.nbanks = int(np.ceil(self.nneurons/8.))
        self.curbank = 0
        self.gain = np.ones(self.nneurons)
        self.scale = np.ones(self.nneurons)
        self.mute = np.ones(self.nneurons)
        self.midi_init()
        self.is_processing = 0
        self.tracks = ['Original', 'Resynthesized']
        self.running = True
        self.start_screen()
        
    def midi_init(self):
        pygame.midi.init()
        devcount = pygame.midi.get_count()
        print('Num of midi devices connected: {0}'.format(devcount))
        for i in range(devcount):
            dev = pygame.midi.get_device_info(i)
            if (dev[1].split()[0] == 'nanoKONTROL2' and
                dev[1].split()[-1] == 'SLIDER/KNOB'):
                self.devid = i
                print("Using {0}".format(dev[1]))
                self.cont =  pygame.midi.Input(self.devid)

    def start_screen(self):
        self.screen = pygame.display.set_mode([640, 480])
        pygame.display.set_caption('Midi Control Window')
        pygame.font.init()
        self.font = pygame.font.SysFont("Parchment", 24)
        self.update_mult()
        self.update_text()

    def update_text(self):
        self.screen.fill((0,0,0))
        text = self.font.render('Neuron',1,(255,255,255))
        self.screen.blit(text, (20,20))
        text = self.font.render('Gain',1,(255,255,255))
        self.screen.blit(text, (230,20))
        text = self.font.render('Scale',1,(255,255,255))
        self.screen.blit(text, (320,20))
        text = self.font.render('Adj. Value',1,(255,255,255))
        self.screen.blit(text, (420,20))
        text = self.font.render('Mute',1,(255,255,255))
        self.screen.blit(text, (520,20))
        for i in range(NEURONS_PER_BANK):
            n = self.curbank * NEURONS_PER_BANK + i
            text = self.font.render('{0}'.format(n), 1, (255,255,255))
            self.screen.blit(text, (20,(i+2)*30))
            text = self.font.render("%.5f" % self.gain[n], 1,
                                    (255,255,255))
            self.screen.blit(text, (220,(i+2)*30))
            text = self.font.render("%.5f" % self.scale[n], 1, (255,255,255))
            self.screen.blit(text, (320,(i+2)*30))
            text = self.font.render("%.5f" % self.encoded[n], 1, 
                                    (255,255,255))
            self.screen.blit(text, (420,(i+2)*30))
            text = self.font.render("M" if not self.mute[n] else "", 1, 
                                    (255,255,255))
            self.screen.blit(text, (520,(i+2)*30))
        text = self.font.render('Queued Track: {0}'.format(
                self.tracks[self.is_processing]), 1, (255,255,255))
        self.screen.blit(text, (20,(NEURONS_PER_BANK+4)*30))
        pygame.display.flip()

    def run(self):
        while self.running:
            pygame.event.pump()
            while self.cont.poll():
                data = self.cont.read(MID_BUF)
                ctrl = data[-1][0][1]
                val = data[-1][0][2]
                #print(ctrl)
                if ctrl == 58 and val == 127: # Track <
                    self.change_bank(-1)
                if ctrl == 59 and val == 127: # Track >
                    self.change_bank(1)
                if ctrl == 46 and val == 127: # cycle button
                    self.exit()
                if ctrl == 60 and val == 127: # set
                    self.reset_all()
                if ctrl == 43  and val == 127: # <<
                    self.toggle_processing()
                if ctrl == 44  and val == 127: # >>
                    self.toggle_processing()
                if ctrl == 42 and val == 127: # stop button
                    self.stop()
                if ctrl == 41 and val == 127: # play button 
                    self.play()
                if ctrl == 45 and val == 127: # record button 
                    self.reset_all()
                if ctrl == 60 and val == 127: # set button 
                    self.mute_all()
                if ctrl >= 0 and ctrl < 8: # faders
                    self.gain[ctrl + NEURONS_PER_BANK*self.curbank
                              ] = val/127.
                    self.update_mult()
                if ctrl >= 16 and ctrl < 24: # knobs
                    self.scale[ctrl-16 + NEURONS_PER_BANK*self.curbank
                               ] = val/127. * 2
                    self.update_mult()
                if ctrl >= 48 and ctrl < 56 and val == 127: # fader mute
                    self.mute_t_neuron(ctrl-48 + NEURONS_PER_BANK*self.curbank)
                if ctrl >= 64 and ctrl <= 71 and val == 127: # fader record
                    self.reset_neuron(ctrl-64 + NEURONS_PER_BANK*self.curbank)

    def change_bank(self, pos):
        self.curbank += pos
        self.curbank %= self.nbanks
        self.update_text()

    def toggle_processing(self):
        self.is_processing += 1
        self.is_processing %= 2
        self.queue.put(['is_processing', self.is_processing])
        self.update_text()

    def reset_all(self):
        self.gain[:] = 1
        self.scale[:] = 1
        self.mute[:] = 1
        self.update_mult()
        self.update_text()

    def mute_all(self):
        self.mute[:] = 0
        self.update_mult()
        self.update_text()

    def mute_t_neuron(self, n):
        self.mute[n] += 1
        self.mute[n] %= 2
        self.update_mult()
        self.update_text()

    def reset_neuron(self, n):
        self.gain[n] = 1
        self.scale[n] = 1
        self.mute[n] = 1
        self.update_mult()
        self.update_text()

    def empty(self):
        while self.cont.poll():
            self.cont.read(MID_BUF)

    def update_mult(self):
        self.encoded = self.gain*self.scale*self.mute
        self.queue.put(["mult", self.encoded])
        self.update_text()

    def play(self):
        self.queue.put(["play_pause", None])

    def stop(self):
        self.queue.put(["stop", None])
        pyplot.close()

    def exit(self):
        pyplot.close()
        pygame.quit()
        self.queue.put(['shutdown', None])
        self.running = False

    ### NOT CHECKED
    ### MOVE TO OTHER CLASS
    def plot(self):
        pyplot.clf()
        pyplot.subplot(2,1,1)
        features.feature_plot(self.R.F.X, dbscale=True, nofig=True, 
                              title_string="Orig")
        if hasattr(self.R.F, 'X_hat'):
            pyplot.subplot(2,1,2)
            features.feature_plot(np.abs(self.R.F.X_hat), dbscale=True, nofig=True, 
                                  title_string="Recon")

class PlayStreaming(object):
    def __init__(self, nfft, wfft, nhop, wav_file, queue, model_file):
        self.nfft = nfft
        self.wfft = wfft
        self.nhop = nhop
        self.nolap = self.nfft-self.nhop
        self.wav_file = wav_file
        self.queue = queue
        self.model_file = model_file
        self.init_model()

        self.win = np.hanning(self.wfft)
        self.buf = np.zeros(self.nhop)
        self.olap_buf = np.zeros(self.nolap)
        self.wf = wave.open(wav_file, 'rb')
        self.p = pyaudio.PyAudio()
        self.is_processing = 0
        rate = self.wf.getframerate()
        channels = self.wf.getnchannels()
        format = self.p.get_format_from_width(self.wf.getsampwidth())
        self.stream = self.p.open(rate=rate,
                             channels=channels,
                             format=format,
                             input=False,
                             output=True,
                             input_device_index=None,
                             output_device_index=None,
                             frames_per_buffer=self.nhop,
                             start=True)
        self.playing = False
        self.run()

    def init_model(self):
        with open(self.model_file, 'r') as f:
            model = cPickle.load(f)
        params = []
        if hasattr(model, "autoencoders"):
            for a in model.autoencoders:
                params.append({})
                params[-1]['act_enc'] = a.act_enc.name if hasattr(
                    model.autoencoders[0].act_enc, 'name') else None
                params[-1]['act_dec'] = a.act_dec.name if hasattr(
                    model.autoencoders[0].act_dec, 'name') else None
                for p in a.get_params():
                    params[-1][p.name] = p.get_value()
            nneurons = model.autoencoders[-1].get_output_space().dim
        else:
            params.append({})
            params[-1]['act_enc'] = model.act_enc.name if hasattr(
                model.act_enc, 'name') else None
            params[-1]['act_dec'] = model.act_dec.name if hasattr(
                model.act_dec, 'name') else None
            for p in model.get_params():
                params[0][p.name] = p.get_value()
            nneurons = model.nhid
        self.params = params
        self.queue.put(['nneurons',nneurons], False)
        self.model = model

    def play_frame(self):
        wf = self.wf
        nfft = self.nfft
        wfft = self.wfft
        nhop = self.nhop
        nolap = self.nolap
        ix  = wf.tell()
        data = wf.readframes(wfft)
        if len(data) < 2*wfft:
            self.wf.rewind()
            self.playing = False
            return
        wf.setpos(ix+nhop)
        data = np.array(struct.unpack("h"*wfft, data)) * self.win
        fft = np.fft.rfft(data, nfft) / nfft
        X = np.abs(fft)
        phase = np.angle(fft)
        if self.is_processing:
            X = self.process_frame(X)
        data = np.real(nfft * np.fft.irfft(X * np.exp(1j * phase)))
        self.buf[:] = data[:nhop]
        self.buf += self.olap_buf[:nhop]
        self.olap_buf = np.r_[self.olap_buf[nhop:], np.zeros(nhop)]
        self.olap_buf += data[nhop:]


        self.buf = np.where(self.buf > np.iinfo('short').max, 
                            np.iinfo('short').max, self.buf)
        self.buf = np.where(self.buf < np.iinfo('short').min, 
                            np.iinfo('short').min, self.buf)

    def process_frame(self, X):
        for p in self.params:
            X = self.activation(X, p['W'], p['hb'], p['act_enc'])
        X *= self.mult
        for p in self.params[::-1]:
            X = self.activation(X, p['Wprime'], p['vb'], p['act_dec'])
        # This is a hack. Need to record the normalization constant of the 
        # training set
        X *= 70
        return X

    @staticmethod
    def activation(X, W, b, a):
        X = np.dot(X, W) + b
        if a is not None:
            X = getattr(PlayStreaming, a)(X)
        return X
        
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def play_stream(self):
        self.playing = True
        while self.playing:
            self.play_frame()
            data = struct.pack("h"*(self.nhop), *self.buf)
            self.stream.write(data, self.nhop)
            self.cmd_parse()

    def run(self):
        while True:
            self.cmd_parse()

    def cmd_parse(self):
        cmd,msg = self.queue.get() if not self.queue.empty() else ['',None]
        if cmd != '':
            #print(cmd)
            pass
        if cmd == "play_pause":
            if self.playing:
                self.playing = False
            else:
                self.play_stream()
        if cmd == "stop":
            self.wf.rewind()
            self.playing = False
        if cmd == "shutdown":
            self.shutdown()
        if cmd == "is_processing":
            self.is_processing = msg
        if cmd == "mult":
            self.mult = msg

    def shutdown(self):
        self.stream.stop_stream()
        self.stream.close()
        self.wf.close()
        self.p.terminate()
        exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Use a MIDI controller to inspect the code layer of a '+ \
            'deep autoencoder. '+\
            'Command: python autocontrol.py modelFile audioFile [--plot] '+\
            'where modelfile is the filename of the model saved by using '+\
            'deepAE.py audioFile is a')
    parser.add_argument('modelFile', type=str,
                        help="a deep autoencoder trained with deepAE.py")
    parser.add_argument('audioFile', type=str,
                        help="an audio file to resynthesize")
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    Q = multiprocessing.Queue()
    P = multiprocessing.Process(target=PlayStreaming, args=(2048, 1025, 512, 
                                                            args.audioFile, Q,
                                                            args.modelFile))
    P.start()
    A = Autocontrol(args.modelFile, args.audioFile, Q, plot=args.plot)
    A.run()
    sys.exit(0)
