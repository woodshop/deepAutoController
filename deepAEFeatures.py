import bregman
from scipy import io
import numpy as np

class Batch():
    def __init__(self, data_dir=None):
        self.data_dir = data_dir
        if self.data_dir is None:
            self.data_dir = '/global/data/casey/sarroff/projects/hamr/data'

    def cqft(self, runid, ncores, debug=False):
        allsongs = np.load(self.data_dir+'/filt_allsongs.npy').item()
        l = len(allsongs)
        if debug:
            l = ncores * 3
        v = allsongs.values()
        alt_dir = '/global/data/casey/sarroff/projects/groove/data'
        for i in range(runid, l, ncores):
            wav_f = alt_dir+'/audio/wav/'+str(v[i].id)+'.wav'
            # read wav file
            #beats = v[i].beat4.T[0]
            print("\t{0}\t{1}".format(float(i)/l, wav_f))
            x = bregman.sound.WavOpen(wav_f, 6*22050).sig[22050:]
            p = bregman.features.Features().default_feature_params()
            p['hi'] = 10000
            p['nfft'] = 2048
            p['nhop'] = 1024
            p['sample_rate'] = 22050
            p['wfft'] = 2048
            F = bregman.features.Features(x, p)
            data_file = ('/scratch/sarroff/{0}/'.format(runid)+
                         v[i].id+'.cqft.npz')
            np.savez(data_file, CQFT=F.CQFT, POWER=F.POWER, Q=F.Q,
                    STFT=F.STFT, feature_params=F.feature_params)

def collect(data_base='/global/data/casey/sarroff/projects/hamr/data'):
    allkeys = np.load(data_base+'/allkeys.npy')
    tmp = np.load(data_base+'/cqft/'+allkeys[0]+'.cqft.npz')['CQFT'].shape
    allcqft = np.empty((len(allkeys), tmp[0]*tmp[1]))
    for i,k in enumerate(allkeys):
        print(float(i)/len(allkeys))
        allcqft[i] = np.load(data_base+'/cqft/'+allkeys[0]+'.cqft.npz'
                             )['CQFT'].flatten()
    return allcqft

if __name__ == "__main__":
    """
    Main function for batch extracting features.
    """
    import sys
    B = Batch()
    if sys.argv[1] == "cqft":
        runid = int(sys.argv[2])
        ncores = int(sys.argv[3])
        print("Extracting cqft... runid={0}, ncores={1}".format(
            runid, ncores))
        if len(sys.argv) == 5 and sys.argv[4] == "True":
            print("Debug on")
            debug = True
        else:
            debug = False
        B.cqft(runid, ncores, debug)
