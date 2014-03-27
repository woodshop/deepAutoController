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
import deepAE
import bregman
from bregman import sound
import pygame
import pygame.midi
import theano
import theano.tensor as T
import numpy as np
import sys
from matplotlib import pyplot
import os

NNEURONS = 8
BUF = 1024
os.environ['SDL_VIDEO_WINDOW_POS'] = "0,400"
class Autocontrol(object):
    def __init__(self, modelFile, audioFile, plot=False):
        self.recon = None
        self.plotting = plot
        self.modelFile = modelFile
        self.audioFile = audioFile
        self.mult = theano.shared(value=np.ones(NNEURONS,
                                             dtype=theano.config.floatX),
                               borrow=True)
        self.model = deepAE.rebuild(modelFile, self.mult)
        # HARD CODED: USE 5 seconds of audio file starting at 1.0 secs
        # WARNING: assumes 22050 sr. Duration of clip will vary based upon
        # actual sr
        w = bregman.sound.WavOpen(audioFile, 6*22050, verbosity=0)
        self.x = w.sig[22050:]
        self.set_f_params(w.sample_rate)
        self.F = bregman.features.Features(self.x, self.p)
        self.x_hat_orig = self.F.inverse(usewin=False)
        self.x_hat_orig -= self.x_hat_orig.min()
        self.x_hat_orig /= np.abs(self.x_hat_orig).max()
        self.CQFT_orig = self.F.CQFT.copy()
        self.model_in = theano.shared(
            value=self.CQFT_orig.flatten().astype(theano.config.floatX),
            borrow=True)
        self.midi_init()
        self.level = np.ones(NNEURONS)
        self.sign = np.ones(NNEURONS)
        self.scale = np.ones(NNEURONS)
        self.start_screen()

    def set_f_params(self, sr):
        # HARD CODED: Same feature params used to train models in deepAE.py
        p = bregman.features.Features().default_feature_params()
        p['hi'] = 10000
        p['nfft'] = 2048
        p['nhop'] = 1024
        p['sample_rate'] = sr
        p['wfft'] = 2048
        self.p = p

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
        self.screen = pygame.display.set_mode([500,500])
        pygame.display.set_caption('Midi Control Window')
        pygame.font.init()
        self.font = pygame.font.SysFont("Parchment", 45)
        self.update_text()

    def update_text(self):
        #print(self.mult.get_value())
        self.screen.fill((0,0,0))
        text = self.font.render('Neuron',1,(255,255,255))
        self.screen.blit(text, (20,20))
        text = self.font.render('Scale',1,(255,255,255))
        self.screen.blit(text, (150,20))
        text = self.font.render('Value',1,(255,255,255))
        self.screen.blit(text, (300,20))
        for i in range(NNEURONS):
            text = self.font.render('{0}'.format(i+1), 1, (255,255,255))
            self.screen.blit(text, (20,(i+2)*30))
            text = self.font.render("%.3f" % self.scale[i], 1, (255,255,255))
            self.screen.blit(text, (150,(i+2)*30))
            text = self.font.render("%.5f" % self.mult.get_value()[i], 1,
                                    (255,255,255))
            self.screen.blit(text, (300,(i+2)*30))
        pygame.display.flip()

    def run(self):
        while True:
            pygame.event.pump()
            if self.cont.poll():
                data = self.cont.read(BUF)
                ctrl = data[-1][0][1]
                val = data[-1][0][2]
                if ctrl == 42 and val == 127: # stop button
                    self.exit()
                    break
                if ctrl == 45 and val == 127: # record button
                    self.synth()
                if ctrl == 41 and val == 127: # play button
                    self.play_orig()
                if ctrl >= 0 and ctrl <= 8: # faders
                    self.level[ctrl] = val/127.
                    self.update_mult()
                if ctrl >= 16 and ctrl <= 24: # knobs
                    self.scale[ctrl-16] = val/127. * 10
                    self.update_mult()
                if ctrl >= 32 and ctrl <= 40: # solo buttons
                    self.sign[ctrl-32] = -1. if val == 127 else 1.
                    self.update_mult()

    def empty(self):
        while self.cont.poll():
            self.cont.read(BUF)

    def update_mult(self):
        self.mult.set_value(self.level*self.scale*self.sign)
        self.update_text()

    def synth(self):
    	# Shape has been hard coded
        self.F.X = self.model.reconstruct_input_ext(self.model_in
                                                    )[-1].eval().reshape(87,
                                                                         -1)
        if self.plotting: self.plot()
        self.F.inverse(V_hat = self.F.X, usewin=False)
        self.F.x_hat -= self.F.x_hat.min()
        self.F.x_hat /= np.abs(self.F.x_hat).max()
        sound.play(self.F.x_hat, self.p['sample_rate'])

    def play_orig(self):
        sound.play(self.x_hat_orig, self.p['sample_rate'])

    def exit(self):
        self.cont.close()
        pygame.quit()

    def plot(self):
        pyplot.clf()
        pyplot.subplot(2,1,1)
        # HARD CODED!
        pyplot.imshow(self.F.X, origin='lower', aspect='auto')
        pyplot.ylabel('Freq Band')
        pyplot.subplot(2,1,2)
        pyplot.imshow(self.CQFT_orig, origin='lower', aspect='auto')
        pyplot.xlabel('Time')
        pyplot.ylabel('Freq Band')


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
    a = Autocontrol(args.modelFile, args.audioFile, plot=args.plot)
    a.run()
    pyplot.close()
    sys.exit(0)
