import os
import matplotlib.pyplot as plt
import numpy as np
from joblib import load
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit


paths = {
    "tracks": "/sdf/group/kipac/users/gammaTPC/DynamicRange/sensor_response",
    "wire_response": "/sdf/group/kipac/users/gammaTPC/DynamicRange/sensor_response/50simCS3_20MHz.npy",
}

def get_wire_signal_tensor(index):
    DATA     = np.load(paths["wire_response"], allow_pickle=True)
    dat      = np.array([i["signals"][index] for i in DATA])
    return dat


# def get_track_path(paths, energy, track_num):
#     track_names = paths["track_dict"][f'E{energy:06.0f}']
#     return os.path.join(paths['tracks'], f'E{energy:06.0f}',
#                         track_names[track_num])

def extrapolate_signal_response(signal_response, num_samples=1000):
    """
    Extrapolates the signal response to the left and concatenates it with the original.
    If the curve fitting fails, it returns zeros for the extrapolated part.

    Parameters:
    signal_response (numpy array): The original signal response data.
    num_samples (int): The number of samples to extrapolate.

    Returns:
    numpy array: The concatenated extrapolated and original signal response.
    """
    x = np.arange(len(signal_response))
    fit_func = lambda x, a, b, x0: a + b / (x - x0)**3
    try:
        popt, _ = curve_fit(fit_func, x, signal_response, p0=[-4000, 1000, 500])
        extrapolated_x = np.arange(-num_samples, 0)
        extrapolated_y = fit_func(extrapolated_x, *popt)
    except Exception as e:
        print(f"Curve fitting failed: {e}")
        extrapolated_y = np.zeros(num_samples)
    return np.concatenate((extrapolated_y, signal_response))



#--------------
class Sliced_track():
    """ Sliced Electron track, inherits from class Track
    by default it is drifted"""

    def __init__(self,electrons,params):
        """
        function:
            in -> sliced electrons and parameters
                electrons: drifted and confined between 0 and w_pitch
            a track in this context is defined as: collection of charges that will
            drift between 4 wires: # <- in the center of the hashtag.

        """

        self.sampling_rateMHz = 20
        self.xy_resolution = params.spatial_resolution['sigma_xy']

        self.wire_pitch    = params.coarse_grids['pitch']
        self.z_spatial_resolution = \
            params.charge_drift['velocity']/(self.sampling_rateMHz*1e6)
        
        self.params = params
        self.electrons = electrons


        # check that all are between 0 and pitch
        if electrons.shape[0] < 4:
            assert all(self.electrons.min(axis=1)[0:2]>=0),\
                "Spillover of charges at 0"
            assert all(self.electrons.max(axis=1)[0:2]<=self.wire_pitch),\
                "Spillover of charges at pitch"

    #------------------------------------------------------
    #                      PIXEL Readout
    #------------------------------------------------------
    def current(self, display=True):
        """
        @bahrudin
        uses histogramdd to voxelize the volume into boxes of
        pixel_pitch X pixel_pitch X z_resolution
        and counts electrons in those boxes.
        That returns x,y,z of boxes, and # of electrons in them
        """

        pixel_pitch = self.params.inputs['pixels']['pitch']
        wires_pitch = self.wire_pitch
        dh = self.z_spatial_resolution
        electrons   = self.electrons

        z_offset_samples = int(min(electrons[2])//dh)
        min_z_d = z_offset_samples * dh
        xdomain = np.arange(0,wires_pitch,pixel_pitch)
        ydomain = np.arange(0,wires_pitch,pixel_pitch)
        zdomain = np.arange(min_z_d,max(electrons[2])+2*dh,dh)
        bins    = [xdomain,ydomain,zdomain]

        if self.electrons.shape[0] == 3:
            H,L = np.histogramdd(self.electrons.T, bins=bins)
            print("electrons have 3 - old Tracks ?!...")
        if self.electrons.shape[0] == 4:
            H,L = np.histogramdd(self.electrons[:3].T,
                                 bins=bins,
                                 weights=self.electrons[-1])
            
        if np.sum(H)==0:
            print("pixels problem, trying different routine")
            r = electrons
            dxy = 20*pixel_pitch/19
            x_ind = np.floor(r[0]/dxy).astype(int)
            y_ind = np.floor(r[1]/dxy).astype(int)
            z_ind = np.floor(r[2]/dh).astype(int) - int(min_z_d//dh)
            combined     = np.array([x_ind,y_ind,z_ind]).T
            struck_chips = np.unique(combined,axis=0)
            for i,j,k in struck_chips:
                mask = (x_ind==i)&(y_ind==j)&(z_ind==k)
                H[i,j,k] = mask.sum()

        total_current = H.sum(axis=(0,1))
        timescale = np.arange(0,len(total_current))/self.sampling_rateMHz
        px,py,pz  = np.unravel_index(H.argmax(), H.shape)

        if display:
            fig, ax = plt.subplots(1, 2,figsize=(12, 6))
            #fig.tight_layout()

            ax[0].grid(linestyle=':',linewidth='1')
            ax[0].step(timescale,total_current)
            ax[0].step(timescale,H[px,py])
            ax[0].set_xlabel('Time [µs]')
            ax[0].set_ylabel('Current e/µs')
            ax[0].set_title(f'Total current from the track')
            ax[0].legend(['Total Signal',
              f"Signal on Central Pixel {px},{py}"])

            ax[1].imshow(H.sum(axis=2),
                         extent=[0,wires_pitch*1e3,0,wires_pitch*1e3])
            ax[1].set_xlabel('X pixels [mm]')
            ax[1].set_ylabel('Y pixels [mm]')
            ax[1].set_title(f'Total current from the track, crossection')

        return H, z_offset_samples

    def signal_on_pixel(self,sampling_rateMHz=20,display=True):
        """
        @bahrudin
        """
        from . import electronics_transfer_functions

        wires_pitch = self.params.inputs['coarse_grids']['pitch']
        I, z_offset_samples = self.current(display=False)
        pixel_response = electronics_transfer_functions.get_pixel_response()
        H = np.apply_along_axis(lambda m: np.convolve(m, pixel_response,
                                                      mode='full'),
                                                      axis=2, arr=I)

        if display:
            total_current = H.sum(axis=(0,1))
            timescale = np.arange(0,len(total_current))/sampling_rateMHz
            px,py,pz =  np.unravel_index(H.argmin(), H.shape)

            fig,ax = plt.subplots(1, 2,figsize=(12, 6))
            #fig.tight_layout()

            ax[0].grid(linestyle=':',linewidth='1')
            ax[0].step(timescale,total_current)
            ax[0].step(timescale,H[px,py])
            ax[0].set_xlabel('Time [µs]')
            ax[0].set_ylabel('Current e/µs')
            ax[0].set_title(f'Total current from the track')
            ax[0].legend(['Total Signal',
                          f"Signal on Central Pixel {px},{py}"])

            ax[1].imshow(H.sum(axis=2),
                         extent=[0,wires_pitch*1e3,0,wires_pitch*1e3])
            ax[1].set_xlabel('X pixels [mm]')
            ax[1].set_ylabel('Y pixels [mm]')
            ax[1].set_title(f'Total current from the track, crossection')

            #D = np.asarray([timescale, H[px//2,py//2]])
            #np.savetxt("foo.txt", D.T, delimiter=",")

        return H, z_offset_samples


    def analog_pixel_readout(self,
                             sampling_rateMHz=20,
                             display=True):
        """
        @bahrudin
        """
        from scipy.signal import bessel, filtfilt

        peaking_time = self.params.inputs['pixels']['peaking_time']
        # Amplification Factor
        N = -1
        filter_order = 5
        highcut = np.pi / peaking_time

        I, z_offset_samples = self.signal_on_pixel(display=False)
        b,a = bessel(filter_order,
                     highcut,
                     btype='lowpass',
                     output='ba',
                     fs=sampling_rateMHz * 1e6)

        analog_signal = np.apply_along_axis(lambda m:filtfilt(b, a, m),
                                            axis=2, arr=I * N)

        if display:
            total_analog_signal = analog_signal.sum(axis=(0,1))
            timescale = np.arange(0,len(total_analog_signal))/sampling_rateMHz
            px,py,pz = np.unravel_index(np.abs(analog_signal).argmax(), analog_signal.shape)

            fig, ax = plt.subplots(1, 1,figsize=(6, 6))
            #fig.tight_layout()

            ax.grid(linestyle=':',linewidth='1')
            ax.step(timescale,total_analog_signal)
            ax.step(timescale,analog_signal[px,py])
            ax.set_xlabel('Time [µs]')
            ax.set_ylabel('Current e/µs')
            ax.set_title(f'Analog Charge Readout {peaking_time*1e6} µs Peaking Time')
            ax.legend(['Total Signal',
                          f"Signal on Maximal Pixel {px},{py}"])

            # D = np.asarray([timescale, analog_signal[px,py]])
            # np.savetxt("analog_signal_1MeV.txt", D.T, delimiter=",")
        return analog_signal, z_offset_samples

    def digital_pixel_readout(self,
                             sampling_rateMHz=20,
                             accumulation = 1,
                             display=True):
        """
        @bahrudin
        """
        from scipy import signal
        peaking_time = self.params.inputs['pixels']['peaking_time']

        digital_sampling_rateMHz = \
            self.params.inputs['pixels']['digital_sampling_freq']/1e6

        A, z_offset_samples = self.analog_pixel_readout(display=False)
        sample_ratio = int(sampling_rateMHz/digital_sampling_rateMHz)
        assert sampling_rateMHz%digital_sampling_rateMHz == 0

        D = A[:,:,::sample_ratio]
        z_off_digital = z_offset_samples/sample_ratio

        if accumulation > 1:
            filter_ = np.ones(accumulation) / accumulation
            D = np.apply_along_axis(lambda m: np.convolve(m, filter_, mode='valid'),
                                    axis=2, arr=D)


        if display:
            resampled_len = len(D[0,0])
            total_digital_signal = D.sum(axis=(0,1))
            timescaleD = np.arange(0,resampled_len)/digital_sampling_rateMHz
            timescaleA = np.arange(0,len(A[0,0]))/sampling_rateMHz
            px,py,pz = np.unravel_index(np.abs(D).argmax(), D.shape)

            fig, ax = plt.subplots(1, 1,figsize=(6, 6))
            #fig.tight_layout()

            ax.grid(linestyle=':',linewidth='1')
            ax.plot(timescaleD,total_digital_signal,'.')
            ax.plot(timescaleA,A.sum(axis=(0,1)),c='silver',alpha=0.7)
            ax.plot(timescaleD,D[px,py],'.')
            ax.plot(timescaleA,A[px,py],'--',c='silver',alpha=0.4)
            ax.set_xlabel('Time [µs]')
            ax.set_ylabel('Current e/µs')
            ax.set_title(f'Digital Charge Readout \n{peaking_time*1e6} µs Peaking Time\n'+\
                         f"{digital_sampling_rateMHz} MHz Digital Sample Rate")
            ax.legend(['Total Digitized Signal', 'Total Analog Signal',
                       f"Digitized Signal, Pixel: {px},{py}",
                       f"Analog Signal, Pixel: {px},{py}"])
        return D, z_off_digital


    def recreate_track(self,
                        display=True,
                        units='mm',
                        set_limits=True):
        """
        @bahrudin
        """

        ENC         = self.params.inputs["pixels"]["noise"]
        threshold   = ENC * self.params.inputs["pixels"]["threshold_sigma"]
        pixel_pitch = self.params.inputs['pixels']['pitch']
        charge_velocity = self.params.charge_drift['velocity']
        digital_sampling_rate = \
            self.params.inputs['pixels']['digital_sampling_freq']
        z_pitch    = charge_velocity/(digital_sampling_rate)

        enable_enc = self.params.enable_w_noise

        scale = 1e3
        if units == 'm':
            scale = 1
        if units == 'cm':
            scale = 1e2
        if units == 'um':
            scale = 1e3

        S=[]
        digital_output, z_off_digital = self.digital_pixel_readout(display=False)

        for coordinate, signalValue in np.ndenumerate( abs(digital_output)):
            charge = signalValue / digital_sampling_rate
            charge+= np.random.normal(scale=ENC) * enable_enc
            # if the charge goes over the threshold on first try
            if charge > threshold:
                x_pos = coordinate[0] * pixel_pitch * scale
                y_pos = coordinate[1] * pixel_pitch * scale
                z_pos = (coordinate[2] + z_off_digital) * z_pitch * scale
                S.append([x_pos,y_pos,z_pos,charge])

        if len(S) < 1:
            accumulation = 4
            print(f"No charges above treshold, trying with accumulation = {accumulation} ")
            digital_output, z_off_digital = \
                self.digital_pixel_readout(display=False, accumulation=accumulation)

            for coordinate, signalValue in np.ndenumerate( abs(digital_output)):
                charge = signalValue / digital_sampling_rate
                charge+= np.random.normal(scale=ENC/np.sqrt(accumulation)) * enable_enc
                # if the charge goes over the threshold on first try
                if charge > threshold / np.sqrt(accumulation):
                    print('Eh eve ga, iznad ove druge granice !!!')
                    x_pos = coordinate[0] * pixel_pitch * scale
                    y_pos = coordinate[1] * pixel_pitch * scale
                    z_pos = (coordinate[2] + z_off_digital) * z_pitch * scale
                    S.append([x_pos,y_pos,z_pos,charge])

        if len(S) < 1:
            print(f"No charges above treshold, ------- ")
            S.append([1e-6,0,0,0])

        x,y,z,c = np.array(S).T

        if display:
            fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(1,1,1,projection='3d')
            ax.scatter(x,y,z,s=700*c/np.max(c))

            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
            ax.set_zlabel('z (mm)')

            if set_limits:
                ax.set_xlim(0,10)
                ax.set_ylim(0,10)

            deposited_charge = len(self.electrons[0])

            ax.set_title(
                f'Part of a track \n'
                + f'{deposited_charge/1000:4.1f}K e- Deposited Charge\n'
                + f'{np.sum(c)/1000:4.1f}K e- Detected Charge\n'
                + f'{pixel_pitch*1e6:4.0f}  $\mu$m pitch,'
                #+ f'{sampling_time*1e9:3.0f}'
                #+ ' ns sampling, '
                #+ '$\sigma_e$ '
                + f' / {threshold:4.1f}  e- ')
        return S

    #-----------------------------------------------------------------
    #                           WIRE READOUT
    #-----------------------------------------------------------------

    def get_current(self,display=True):

        dh  = self.z_spatial_resolution
        dxy = self.xy_resolution
        xdomain = np.arange(0,self.wire_pitch, dxy)
        ydomain = np.arange(0,self.wire_pitch, dxy)
        zdomain = np.arange(0,self.electrons.max(axis=1)[2]+2*dh,dh)


        bins=[xdomain,ydomain,zdomain]
        if self.electrons.shape[0] == 3:
            H,L = np.histogramdd(self.electrons.T, bins=bins)
        if self.electrons.shape[0] == 4:
            H,L = np.histogramdd(self.electrons[:3].T,
                                 bins=bins,
                                 weights=self.electrons[-1])

        # if the electrons are all in one voxel (below resolution)
        # the histogramnn will fail.
        # below is an alternative way of doing it
        if np.sum(H)==0:
            print("problem, trying different routine")
            r = self.electrons
            dxy = dxy*20/19
            x_ind = np.floor(r[0]/dxy).astype(int)
            y_ind = np.floor(r[1]/dxy).astype(int)
            z_ind = np.floor(r[2]/dh).astype(int)
            combined     = np.array([x_ind,y_ind,z_ind]).T
            struck_chips = np.unique(combined,axis=0)
            for i,j,k in struck_chips:
                mask = (x_ind==i)&(y_ind==j)&(z_ind==k)
                H[i,j,k] = mask.sum()

        total_current = H.sum(axis=(0,1))
        timescale = np.arange(0,len(total_current))/self.sampling_rateMHz

        if display:
            fig, ax = plt.subplots(1, 2,figsize=(12, 6))
            #fig.tight_layout()

            ax[0].grid(linestyle=':',linewidth='1')
            ax[0].plot(timescale,total_current,'.')
            ax[0].set_xlabel('Time [µs]')
            ax[0].set_ylabel('Current e/µs')
            ax[0].set_title(f'Total current from the track')

            ax[1].imshow(H.sum(axis=2),extent=[0,10,0,10])
            ax[1].set_xlabel('X location [mm]')
            ax[1].set_ylabel('Y location [mm]')
            ax[1].set_title(f'Total current from the track, crossection')
        return H

    def get_signal_on_wires(self,
                              display=True,
                              apply_filter=False,
                              output = 'analog',
                              sampling_rateMHz=20,
                              extrapolated=False):
        """
        @bahrudin
        """

        from scipy.signal import bessel, filtfilt
        peaking_time = self.params.inputs['coarse_grids']['peaking_time']

        # Amplification Factor
        N = 1
        filter_order = 5
        highcut = np.pi / peaking_time
        b,a = bessel(filter_order,
                     highcut,
                     btype='lowpass',
                     output='ba',
                     fs=sampling_rateMHz * 1e6)

        digital_sampling_rateMHz = \
            self.params.inputs['coarse_grids']['digital_sampling_freq']/1e6

        sample_ratio = int(sampling_rateMHz/digital_sampling_rateMHz)
        assert sampling_rateMHz%digital_sampling_rateMHz == 0

        wire_response = {}
        wire_response["x1"] = get_wire_signal_tensor(0)
        wire_response["x2"] = get_wire_signal_tensor(1)
        wire_response["y1"] = get_wire_signal_tensor(2)
        wire_response["y2"] = get_wire_signal_tensor(3)
        I = self.get_current(display = False).reshape((19*19,-1))
 
        for wire in ["x1","x2","y1","y2"]:
            temp_sig = 0
            for i in range(len(I)):
                if I[i].sum()!=0:
                    current_on_pixel = I[i]       
                    sig_len = len(current_on_pixel)
                    signal_response_on_pixel = \
                        wire_response[wire][i]#[-sig_len:]
                    
                    if extrapolated:
                        signal_response_on_pixel = \
                            extrapolate_signal_response(signal_response_on_pixel,
                                                        num_samples=700)

                    result = np.convolve(current_on_pixel,
                                         signal_response_on_pixel,
                                         mode="full")
                    temp_sig += result
                    

            wire_response[f"signal_on_{wire}"] = temp_sig

        x1 = wire_response["signal_on_x1"]
        x2 = wire_response["signal_on_x2"]
        y1 = wire_response["signal_on_y1"]
        y2 = wire_response["signal_on_y2"]
        


        # adds 30 samples to the end
        zeros = np.zeros(30)
        x1 = np.concatenate((x1,zeros))
        x2 = np.concatenate((x2,zeros))
        y1 = np.concatenate((y1,zeros))
        y2 = np.concatenate((y2,zeros))

        if apply_filter:
            x1 = filtfilt(b, a, x1)
            x2 = filtfilt(b, a, x2)
            y1 = filtfilt(b, a, y1)
            y2 = filtfilt(b, a, y2)



        #digital
        x1d = x1[::sample_ratio]
        x2d = x2[::sample_ratio]
        y1d = y1[::sample_ratio]
        y2d = y2[::sample_ratio]
        return np.array([x1d, x2d, y1d, y2d])


class Event_Processor():
    """
    Processes a big event using wire and pixel readout
    """
    def __init__(self, track, x0, y0, z0):
        """
        function:
            in -> track and global x.y.z offset
        """
        from . import sims_tools
        from . import readout_tools


        cell_bounds = get_cell_bounds(track, 'GAMPixG', 'default')

        sims_params = sims_tools.Params(
            inputs_source='simple_cell',
            cell_bounds=cell_bounds)

        read_params = readout_tools.Params(
            charge_readout_name='GAMPixG',
            cells=sims_params.cells,
        )
        read_params.calculate()

        self.params = read_params
        self.truth = track.truth
        self.wire_pitch = read_params.coarse_grids['pitch']
        self.CELL_HEIGHT = 0.25
        self.velocity = read_params.charge_drift['velocity']
        self.time_frame = self.CELL_HEIGHT / self.velocity
        self.dig_samp_r = read_params.coarse_grids['digital_sampling_freq']
        self.MAX_SAMPLES = int(np.floor(self.time_frame * self.dig_samp_r))
        self.ENC = read_params.coarse_grids['noise']
        self.z_pitch = self.velocity / self.dig_samp_r

        self.max_dist_from_cluster = self.wire_pitch * 5
        
        self.params.enable_w_noise = True
        
        self.wire_count = 20
        self.WIRE_SIGNALS = {'x':np.zeros((self.wire_count+1, self.MAX_SAMPLES)),
                             'y':np.zeros((self.wire_count+1, self.MAX_SAMPLES))}

        if self.params.enable_w_noise:
            self.add_wire_noise()

        #Wire signals structure
        self.M = np.zeros((self.wire_count-1,
                           self.wire_count-1,
                           self.MAX_SAMPLES))

        # trigger signals structure
        self.TG = np.zeros((self.wire_count-1,
                            self.wire_count-1,
                            self.MAX_SAMPLES))

        # ensure min(x,y,z) = 0
        #track.raw['r'] -= np.min(track.raw['r'], axis=1, keepdims=True)

        # check if drifted is vars(track)

        if 'drifted' in vars(track):
            drifted_r = track.drifted['r'] 
            # set the drifted_r so that min_z is at 0
            drifted_r[2] -= np.min(drifted_r[2])
            drifted_r[1] -= np.min(drifted_r[1])
            drifted_r[0] -= np.min(drifted_r[0])
            
            self.drifted_r = drifted_r + np.array([[x0], [y0], [z0]])
            self.drifted_e = track.drifted['num_e']
        else:
            # offset by x, y, z
            track.raw['r'] += np.array([[x0], [y0], [z0]])
            track.apply_drift(depth=z0)
            self.drifted_r = track.drifted["r"]
            self.drifted_e = track.drifted["num_e"]

        self.get_chip_locations_and_charges()
        self.get_signals_on_wires()
        self.get_triggered_wires_and_triggers()
        self.get_activation_coordinates()
        self.get_charges_from_pixels()
        self.cluster_finder()

        #re-run the clusters to get accurate charge
        #self.get_accurate_charge_from_wires()

    # ----------- FOR TESTING PURPOSES ONLY --------------
    def fake_double_track(self,dx=0.1,dy=0.1,dz=0.1):
        """ Clone the charge to a shifted location. Used for test purposes"""
        track.raw['r'] = np.concatenate((track.raw['r'],
                                              track.raw['r']\
                                           +np.array([[dx,dy,dz]]).T),axis=1)


    # ------------------------------------------------------
    #                      WIRE NOISE
    # ------------------------------------------------------

    def add_wire_noise(self):
        def noise(size):
            from scipy.signal import bessel, filtfilt

            sampling_rateMHz = 20
            peaking_time = self.params.coarse_grids['peaking_time']
            highcut = np.pi / peaking_time
            b, a = bessel(
                N=5,
                Wn=highcut,
                btype='lowpass',
                output='ba',
                fs=sampling_rateMHz * 1e6
            )

            digital_sampling_rateMHz = self.dig_samp_r / 1e6
            sample_ratio = int(sampling_rateMHz / digital_sampling_rateMHz)
            assert sampling_rateMHz % digital_sampling_rateMHz == 0

            ENCscale = self.ENC * 5 / peaking_time
            pure_noise = np.random.normal(
                scale=ENCscale,
                size=(size[0], size[1] * sample_ratio)
            )
            w_noise = filtfilt(b, a, pure_noise)[:, ::sample_ratio]
            w_noise[:, 0] = 0
            return w_noise

        size = self.WIRE_SIGNALS['x'].shape
        self.WIRE_SIGNALS['x'] += noise(size)
        self.WIRE_SIGNALS['y'] += noise(size)



    def get_chip_locations_and_charges(self):
        """
        Calculates the unique locations and charges of struck chips
        in the drifted event.
        Returns:
        chip_locations: numpy array
            Unique x and y indices of the struck chips.
        chip_charges: numpy array
            Charges of the struck chips.
        """
        r     = self.drifted_r
        num_e = self.drifted_e 
        r = np.vstack([r, num_e.reshape(1, -1)])

        x_ind = np.floor(r[0] / self.wire_pitch).astype(int)
        y_ind = np.floor(r[1] / self.wire_pitch).astype(int)

        assert x_ind.max() < self.wire_count, 'Too far in X'
        assert y_ind.max() < self.wire_count, 'Too far in Y'

        combined     = np.array([x_ind,y_ind]).T
        struck_chips = np.unique(combined,axis=0)

        #build up voxels: reduce by i*wire_pitch
        chip_charges = []
        for i,j in struck_chips:
            mask = (x_ind==i)&(y_ind==j)
            chip_offset = np.array([[i*self.wire_pitch,
                                     j*self.wire_pitch,
                                     0,0]]).T
            chip_charges.append(r[:,mask] - chip_offset)

        self.chip_locations = struck_chips
        self.chip_charges   = chip_charges
        return 1


    def get_signals_on_wires(self):
        """
        INPUT: activated chips and charges falling on them
        -> Adds up contributions from charges to the signal on each wire
        """
        def apa(a,b,sig_len=350):
            la,lb = len(a),len(b)
            pa,pb = max(lb-la,0),max(la-lb,0)
            return np.pad(a,(0,pa))[:sig_len] + np.pad(b,(0,pb))[:sig_len]

        for k,(i,j) in enumerate(self.chip_locations):
            subtrack = Sliced_track(self.chip_charges[k],self.params)
            w_signal = subtrack.get_signal_on_wires(display=0)
            print("shape of wire signal", w_signal.shape,i,j)

            self.WIRE_SIGNALS['x'][i]   = apa(self.WIRE_SIGNALS['x'][i],
                                              w_signal[0],
                                              self.MAX_SAMPLES)

            self.WIRE_SIGNALS['x'][i+1] = apa(self.WIRE_SIGNALS['x'][i+1],
                                              w_signal[1],
                                              self.MAX_SAMPLES)

            self.WIRE_SIGNALS['y'][j]   = apa(self.WIRE_SIGNALS['y'][j],
                                              w_signal[2],
                                              self.MAX_SAMPLES)

            self.WIRE_SIGNALS['y'][j+1] = apa(self.WIRE_SIGNALS['y'][j+1],
                                              w_signal[3],
                                              self.MAX_SAMPLES)



    #### TRIGGERING STUFF ###
    # New triger design_______________________________
    def primed_trigger(self,x1,x2):
        """
        A trigger for yTPC.
        Input::: left and right wire.
        Output::: A list [len(x1)] of 0 and 1.
        *treats X and Y dimensions seperatly

        There are 2 things to consider:
            1. the x1 + x2 signal
            2. each signal seperatly
        """
        sampling_rateMHz = 1
        wire_pitch = self.params.inputs["coarse_grids"]["pitch"]
        wire_height= wire_pitch
        charge_vel = self.params.charge_drift['velocity']
        dz         = charge_vel / (sampling_rateMHz*1e6)
        fall_time  = wire_height/ charge_vel

        threshold = self.params.inputs["coarse_grids"]["threshold_sigma"]
        threshold*= self.params.inputs["coarse_grids"]["noise"]
        sig_thres = threshold * charge_vel/ wire_height * 2
        #sig_thres = 17_600 * threshold

        # max negativity occurs at ~ wire_height/2 above the wire
        ps = int(np.floor( wire_height/(2*dz) ) + 1)

        def persist_activation(a,persistance_samples=ps):
            persistance = np.zeros(2*persistance_samples + 1)
            persistance[persistance_samples:] = 1
            tg = np.sign(np.convolve(a,persistance,mode='same')).astype(int)
            return tg

        # detect prime signals
        sig_sum = x1 + x2
        prime1 = persist_activation( (x1 < -sig_thres).astype(int) )
        prime2 = persist_activation( (x2 < -sig_thres).astype(int) )
        sum_prime = sig_sum < -sig_thres*np.sqrt(2)
        prime_sum = persist_activation(sum_prime).astype(int)

        # detect above zero signals
        above_zero1 = (x1 > 0).astype(int)
        above_zero2 = (x2 > 0).astype(int)
        above_zeroT = (sig_sum > 0).astype(int)

        # persist activations
        tg_persistance = int(np.floor( wire_height/dz ) + 1)
        tg1 = persist_activation(above_zero1 * prime1, tg_persistance)
        tg2 = persist_activation(above_zero2 * prime2, tg_persistance)
        tgT = persist_activation(above_zeroT * prime_sum, tg_persistance)

        above_sqrt2sigmaT = (sig_sum > sig_thres*np.sqrt(2)).astype(int)
        post_tg = persist_activation(above_sqrt2sigmaT, 2* tg_persistance)
        # determine if any trigger exists
        any_triger = ((tg1 + tg2 + tgT + post_tg) > 0 ).astype(int)
        D = {}
        D['neg_pos_left']  = (tg1>0).astype(int)
        D['neg_pos_right'] = (tg2>0).astype(int)
        D['neg_pos_sum']   = (tgT>0).astype(int)
        D['only_pos_sum']  = (post_tg>0).astype(int)
        D['any'] = any_triger
        return D

    def get_triggered_wires_and_triggers(self, trigger_kind = 'any'):
        pairwiseXtriggers = []
        pairwiseYtriggers = []

        for i in range(self.wire_count-1):
            sigX_left  = self.WIRE_SIGNALS['x'][i]
            sigX_right = self.WIRE_SIGNALS['x'][i+1]
            tg = self.primed_trigger(sigX_left, sigX_right)
            pairwiseXtriggers.append(tg[trigger_kind])

        for j in range(self.wire_count-1):
            sigY_left  = self.WIRE_SIGNALS['y'][j]
            sigY_right = self.WIRE_SIGNALS['y'][j+1]
            tg = self.primed_trigger(sigY_left, sigY_right)
            pairwiseYtriggers.append(tg[trigger_kind])

        for i in range(self.wire_count-1):
            for j in range(self.wire_count-1):
                self.TG[i,j] = pairwiseXtriggers[i] * pairwiseYtriggers[j]

    def rq_from_wire_signals(self,x,y,z,min_z):
        lx = self.WIRE_SIGNALS['x'][x][z:min_z]
        rx = self.WIRE_SIGNALS['x'][x+1][z:min_z]
        ly = self.WIRE_SIGNALS['y'][y][z:min_z]
        ry = self.WIRE_SIGNALS['y'][y+1][z:min_z]

        def find_max_or_min(a):
            return max(a) if max(a) > abs(min(a)) else min(a)

        sig_sum = lx+rx+ly+ry
        max_val = np.max(sig_sum)
        x_dif = find_max_or_min(lx-rx)
        y_dif = find_max_or_min(ly-ry)

        #print(x,y,x_dif/max_val,y_dif/max_val)

    def get_activation_coordinates(self):
        """
        This function retrieves the coordinates of the activated voxels in
        3D space. It does this by computing the differences in the triggering
        events along the z-axis, determining the onset and offset of the
        activations, and then finding the corresponding end points for each
        activation. The result is a list of coordinates for all activated voxels.
        """
        activated_voxels = []
        changes = np.diff(self.TG,axis=2)
        onset   = np.array(np.where(changes==1)).T
        offset  = np.array(np.where(changes==-1)).T
        max_zf  = changes.shape[-1]

        for x,y,z in onset:
            possible_z = []
            for xf,yf,zf in offset:
                if [x,y]==[xf,yf]:
                    if zf>z:
                        possible_z.append(zf)
            if len(possible_z)==0:
                possible_z.append(max_zf)
            D = {}
            D['x_pix'] = x
            D['y_pix'] = y
            D['z_beg'] = z
            D['z_end'] = min(possible_z)
            # TODO [z:min(possible_z)]
            # In addition to finding the coordinates of triggering
            # It also gets the maxima of the sum and difference of wire_signals
            self.rq_from_wire_signals(x,y,z,min(possible_z))

            activated_voxels.append(D)
        self.activated_voxels = activated_voxels
        return activated_voxels

    def get_charges_from_pixels(self):
        """
        This function retrieves the charges from the pixels that are activated
        during the event. It does this by checking the activated voxels and
        determining the corresponding charges from the chip locations. The
        result is a list of detected charges on the pixels, along with their
        locations and other relevant information.
        """
        detected_charges_on_pixels = []
        D = []

        for av in self.activated_voxels:
            wx,wy = av['x_pix'],av['y_pix']

            # the extra numbers are to offset artifacts from triggering
            beg = av['z_beg'] - 65
            end = av['z_end'] - 10

            z_min = beg * self.z_pitch
            z_max = end * self.z_pitch

            is_there = np.all([wx, wy] == self.chip_locations, axis=1)
            if np.any(is_there):
                IND = np.argwhere(is_there)[0][0]
                charges  = self.chip_charges[IND]
                in_range = (z_min < charges[2]) & (charges[2] < z_max)

                on_pixel = charges[:,in_range]
                if len(on_pixel[0]) > 0:
                    print("signal shape on pixel:",on_pixel.shape)
                    subtrack = Sliced_track(on_pixel,self.params)
                    p_signal = subtrack.recreate_track(display=0,units='m')
                    if np.sum(p_signal,axis=0)[-1]<1:
                        print("WARNING :::: empty signal", p_signal)
                        continue

                    #now adjust for global x y. z is already good
                    globX = wx * self.wire_pitch
                    globY = wy * self.wire_pitch

                    # now add together
                    this_charge = np.array(p_signal)\
                        + np.array([globX,globY,0,0]).T

                    detected_charges_on_pixels.append(this_charge)

                    info = {"wx":wx,
                            "wy":wy,
                            "z_on":z_min,
                            "z_off":z_max,
                            "r":this_charge,
                            }
                    D.append(info)

        assert len(detected_charges_on_pixels)>0


        self.total_charges_on_pixels = np.concatenate(detected_charges_on_pixels)
        self.charges_on_pixels_with_locations = D
        return 1

    def cluster_finder(self):
        from sklearn.cluster import DBSCAN

        max_dist = self.max_dist_from_cluster
        model = DBSCAN(eps=max_dist, min_samples=1)

        data = self.total_charges_on_pixels
        pred = model.fit_predict(data[:,:3])

        # filter out the clusters with <1 charge
        for lbl in np.unique(model.labels_):
            if np.sum(data[model.labels_==lbl],axis=0)[-1] < 1:
                print('Destroying a fake cluster---')
                model.labels_[model.labels_==lbl] = -1

        self.cluster_labels = model.labels_
        n_clusters_ = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
        if n_clusters_ > 1:
            print(f"++  WE HAVE {n_clusters_} clusters  ++")

        return model.labels_


    #------------- Displaying stuff ----------------#
    def display_original_event(self):
        aaa = self.drifted_event['r']
        H,L = np.histogramdd(aaa.T,bins=30)

        S=[]
        for coordinate, charge in np.ndenumerate(H):
            x_pos = coordinate[0]*self.params.inputs['pixels']['pitch']
            y_pos = coordinate[1]*self.params.inputs['pixels']['pitch']
            z_pos = coordinate[2]*self.params.inputs['pixels']['pitch']
            S.append([x_pos,y_pos,z_pos,charge])

        x,y,z,ch = np.array(S).T
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(1,1,1,projection='3d')
        ax.scatter(x,y,z,s=100*ch/np.max(ch))
        ax.set_xlabel('x ')
        ax.set_ylabel('y ')
        ax.set_zlabel('z ')

    def display_total_signals(self,wire=None):
        wx = np.array(self.WIRE_SIGNALS['x'])
        wy = np.array(self.WIRE_SIGNALS['y'])

        if wire!=None:
            wx = np.array(self.WIRE_SIGNALS['x'][wire:wire+2])
            wy = np.array(self.WIRE_SIGNALS['y'][wire:wire+2])

        total = wx.sum(axis=0) + wy.sum(axis=0)

        plt.plot(wx.sum(axis=0))
        plt.plot(wy.sum(axis=0))
        plt.ylabel("Signal [e/sample]")
        plt.xlabel("Sample")
        plt.plot(total)
        plt.legend(["X wires", "Y wires","Total"])

    def display_grid_signals_3D(self, with_2d_heatmap=1):
        S = []
        for coordinate, signalValue in np.ndenumerate(self.M):
            if signalValue > 0:
                x_pos = coordinate[0]*self.wire_pitch
                y_pos = coordinate[1]*self.wire_pitch
                z_pos = coordinate[2]*self.z_pitch
                S.append([x_pos,y_pos,z_pos,signalValue])

        assert len(S)>0, "No charges above treshold"
        S = np.array(S)
        x,y,z,c = S.T

        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(1,1,1,projection='3d')
        ax.scatter(x,y,z,s=100*c/np.max(c))

        if with_2d_heatmap:
            xx, yy = np.meshgrid(np.linspace(0,self.wire_count,
                                             self.wire_count) * self.wire_pitch,
                                 np.linspace(0,self.wire_count,
                                             self.wire_count) * self.wire_pitch)
            pixel_image=np.zeros((self.wire_count,self.wire_count))
            for s1,s2,s3,cha in S:
                s1 = int(np.floor(s1/self.wire_pitch))
                s2 = int(np.floor(s2/self.wire_pitch))
                pixel_image[s2,s1] += cha

            Zdist = 0* np.ones((len(xx),len(yy)))/3
            Color = pixel_image/pixel_image.max()
            ax.plot_surface(xx, yy, Zdist,
                            rstride=1, cstride=1,
                            facecolors=plt.cm.Reds(Color),
                            shade=False, alpha=0.8)
            m = plt.cm.ScalarMappable(cmap=plt.cm.Reds)
            m.set_array(Color*pixel_image.max())
            fig.colorbar(m,fraction=0.046, pad=0.04)

    def display_grid_triggers_3D(self):
        S = []
        for coordinate, signalValue in np.ndenumerate(self.TG):
            if signalValue:
                x_pos = coordinate[0]*self.wire_pitch
                y_pos = coordinate[1]*self.wire_pitch
                z_pos = coordinate[2]*self.z_pitch
                S.append([x_pos,y_pos,z_pos,1])

        assert len(S)>0, "No charges above treshold"
        x,y,z,c = np.array(S).T
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(1,1,1,projection='3d')
        ax.scatter(x,y,z,s=50,marker="s",alpha=0.5)

    def display_triggers_2D(self):
        a,b,c = self.TG.shape
        plt.figure(figsize=(9,5))
        plt.imshow(self.TG.reshape((a*b,c)).T)
        plt.set_cmap('cubehelix')
        plt.xlabel('Pixel ID')
        plt.ylabel('Time sample')
        eff = self.TG.sum()/(a*b*c) * 100
        plt.title(f"Activation of Pixels, activated {np.round(eff,1)}% of time")

    def compare_true_readout(self):
        D = self.activated_voxels
        s = self.chip_locations
        plt.figure(figsize=(8,8))
        d = np.array([[k['x_pix'],k['y_pix']] for k in D])
        plt.ylim((-1,self.wire_count))
        plt.xlim((-1,self.wire_count))
        plt.scatter(*s.T,alpha=0.8,marker='s')
        plt.scatter(*d.T,marker='+')
        plt.legend(["Actual","Wire Activated"])
        plt.xlabel("Wire Number, X")
        plt.ylabel("Wire Number, Y")

    def display_whole_event(self, set_limits=True):
        x,y,z,c  = self.total_charges_on_pixels.T
        clusters = np.array(self.cluster_labels)
        clusters = clusters - clusters.min()
        clusters = clusters/(clusters.max() + 1)
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(1,1,1,projection='3d')
        ax.scatter(x, y, z, s=70*c/np.max(c), c=clusters,cmap="hsv")
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('z (m)')

        if set_limits:
            ax.set_ylim([0, self.wire_count * self.wire_pitch])
            ax.set_xlim([0, self.wire_count * self.wire_pitch])
            ax.set_zlim([0, self.CELL_HEIGHT])

        TITLE = "Dual Scale Charge Readout: Coarse Grids and Pixels\n"
        TITLE+= f'Track Energy {self.truth["track_energy"]:4.0f} keV,\n'
        TITLE+= f'{self.truth["num_electrons"]/1000:4.2f}K e- Deposited Charge\n'
        TITLE+= f'{self.total_charge_from_wires/1e3:4.2f}K e-'
        TITLE+= f'Detected Charge on Pixels with Wires\n'
        ax.set_title(TITLE)

    def display_subtracks(self):
        clusters = np.array(self.cluster_labels)

        for cluster_num,cl in enumerate(self.clustered_charges):
            fig = plt.figure(cluster_num+10, figsize=(10,8))
            ax = fig.add_subplot(1,1,1,projection='3d')

            x,y,z,c  = cl['r']
            dminmaxX = max(x) - min(x)
            dminmaxY = max(y) - min(y)
            dminmaxZ = max(z) - min(z)
            sminmaxX = (max(x) + min(x))/2
            sminmaxY = (max(y) + min(y))/2
            sminmaxZ = (max(z) + min(z))/2

            maxmax = max([dminmaxX,dminmaxY,dminmaxZ])

            ax.scatter(x, y, z, s=120*c/np.max(c))
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_zlabel('z (m)')

            ax.set_xlim([sminmaxX - maxmax/2, sminmaxX + maxmax/2])
            ax.set_ylim([sminmaxY - maxmax/2, sminmaxY + maxmax/2])
            ax.set_zlim([sminmaxZ - maxmax/2, sminmaxZ + maxmax/2])

            TITLE = "Dual Scale Charge Readout: Coarse Grids and Pixels\n"
            TITLE+= f'Cluster:{cluster_num} \n {cl["wire_q"]/1000:4.2f}K e- Deposited Charge\n'
            ax.set_title(TITLE)

    #--------------------------------------------------------------------------
    #this creates the signals on wires from the reconstructed charges
    def add_pad(self,a,b,sig_len=350):
        """Helper function to add two signals and pad them"""
        la,lb = len(a),len(b)
        pa,pb = max(lb-la,0),max(la-lb,0)
        return np.pad(a,(0,pa))[:sig_len] + np.pad(b,(0,pb))[:sig_len]

    def sc_and_pws_from_cluster(self, cluster_num):
        clusters = np.array(self.cluster_labels)

        print('\n------- Cluster:',cluster_num)
        if -1 in clusters:
            print("we have no_class charges")

        cmask = (clusters == cluster_num)
        r = self.total_charges_on_pixels[cmask].T

        if sum(r[3])==0:
            print("No detected charges",r)
            return False

        x_ind = np.floor(r[0] / self.wire_pitch).astype(int)
        y_ind = np.floor(r[1] / self.wire_pitch).astype(int)

        assert x_ind.max() < self.wire_count, 'Too far in X'
        assert y_ind.max() < self.wire_count, 'Too far in Y'

        combined     = np.array([x_ind,y_ind]).T
        struck_chips = np.unique(combined,axis=0)

        chip_charges = []
        reconstructed_z_offset = self.velocity * 3.6e-6
        for i,j in struck_chips:
            mask = (x_ind==i)&(y_ind==j)
            chip_offset = np.array([[i*self.wire_pitch,
                                     j*self.wire_pitch,
                                     reconstructed_z_offset, 0]]).T
            q_here = r[:,mask] - chip_offset
            chip_charges.append(q_here)

        pixW_SIGNALS = {'x':np.zeros((self.wire_count+1, self.MAX_SAMPLES)),
                        'y':np.zeros((self.wire_count+1, self.MAX_SAMPLES))}
        for k,(i,j) in enumerate(struck_chips):
            subtrack = Sliced_track(chip_charges[k],self.params)
            w_signal = subtrack.get_signal_on_wires(display=0)

            pixW_SIGNALS['x'][i]   = self.add_pad(pixW_SIGNALS['x'][i],
                                              w_signal[0],
                                              self.MAX_SAMPLES)

            pixW_SIGNALS['x'][i+1] = self.add_pad(pixW_SIGNALS['x'][i+1],
                                              w_signal[1],
                                              self.MAX_SAMPLES)

            pixW_SIGNALS['y'][j]   = self.add_pad(pixW_SIGNALS['y'][j],
                                              w_signal[2],
                                              self.MAX_SAMPLES)

            pixW_SIGNALS['y'][j+1] = self.add_pad(pixW_SIGNALS['y'][j+1],
                                              w_signal[3],
                                                  self.MAX_SAMPLES)

        return struck_chips, pixW_SIGNALS



    def determine_multiplier(self, px_det, wr_det, PLOT=1):
        PX = np.abs(np.fft.fft(px_det))
        WR = np.abs(np.fft.fft(wr_det))

        max_pixel = max(PX)
        NUM = sum( (PX > max_pixel/7) )//2
        DIV = np.mean(WR[1:NUM]/PX[1:NUM])

        if PLOT:
            plt.figure(np.random.randint(0,1000))
            plt.subplot(211)
            plt.plot(PX)
            plt.plot(WR)
            plt.legend(["pixels",'wires'])
            plt.plot(PX,'.')

            plt.subplot(212)
            ran = np.arange(len(wr_det))
            plt.step(ran, DIV *px_det)
            plt.step(ran, wr_det)
            plt.legend(["pixels",'wires'])
            plt.suptitle(f"Multiplier is {DIV:2.3f}")
            plt.suptitle(f"Taking into account {NUM}")
        print(f"Multiplier is {DIV:2.3f}")
        return DIV

# new one, using math

    def determine_multiplier2(self, px_det, wr_det, PLOT=1):
        #DIV = sum(px_det * wr_det) / sum(px_det * px_det)

        PX = np.abs(np.fft.fft(px_det))
        WR = np.abs(np.fft.fft(wr_det))

        max_pixel = max(PX)
        NUM = sum( (PX > max_pixel/15) )//2
        DIV = np.sum( PX[1:NUM]*WR[1:NUM] ) / np.sum(PX[1:NUM] * PX[1:NUM])

        w_noise = DIV*px_det - wr_det
        dq = w_noise.std()*self.params.inputs['coarse_grids']['peaking_time']

        if PLOT:
            plt.figure(figsize=(8, 6))
            ran = np.arange(len(wr_det))
            plt.step(ran, DIV * px_det, where='mid', linewidth=1.5, color='C0')
            plt.step(ran, wr_det, where='mid', linewidth=1.5, color='C1')
            plt.legend(["Pixels", "Wires"], fontsize=14)
            plt.xlabel("Step number", fontsize=14)
            plt.ylabel("Signal amplitude (V)", fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.title("Step response of detector system", fontsize=16)
            plt.text(0.1, 0.6, f"Multiplier = {DIV:.3f}",
                     transform=plt.gca().transAxes, fontsize=14)
            plt.text(0.1, 0.55, f"STD of ΔSignal = {dq:.1f} e",
                     transform=plt.gca().transAxes, fontsize=14)

            plt.show()
        return DIV


    def get_multiplier_for_cluster(self, struck_chips, pixW_SIGNALS):
        wr_detX = np.zeros(10)
        px_detX = np.zeros(10)
        # need to find the X wires used in this process
        unique_x = np.unique(struck_chips.T[0])
        unique_x = np.append(unique_x,max(unique_x)+1)
        for x_wires in unique_x:
            px_detX = self.add_pad(px_detX,
                                   pixW_SIGNALS['x'][x_wires],
                                   self.MAX_SAMPLES)
            wr_detX = self.add_pad(wr_detX,
                                   self.WIRE_SIGNALS['x'][x_wires],
                                   self.MAX_SAMPLES)

        wr_detY = np.zeros(10)
        px_detY = np.zeros(10)
        unique_y = np.unique(struck_chips.T[1])
        unique_y = np.append(unique_y,max(unique_y)+1)
        for y_wires in unique_y:
            px_detY = self.add_pad(px_detY,
                                   pixW_SIGNALS['y'][y_wires],
                                   self.MAX_SAMPLES)
            wr_detY = self.add_pad(wr_detY,
                                   self.WIRE_SIGNALS['y'][y_wires],
                                   self.MAX_SAMPLES)

        total_from_pixels = self.add_pad(px_detX, px_detY, self.MAX_SAMPLES)

        #only care for where the signal is non zero
        signal_mask = abs(total_from_pixels) > 1e1
        kernel = np.ones(5)/5
        signal_mask = np.convolve(signal_mask, kernel, mode='same').astype(bool)

        total_from_wires = self.add_pad(wr_detX, wr_detY, self.MAX_SAMPLES)
        #total_from_wires[~signal_mask] = 0

        multiplier = self.determine_multiplier2(total_from_pixels[signal_mask],
                                               total_from_wires[signal_mask],
                                               PLOT=1)

        px_sig = total_from_pixels[signal_mask]**2
        err = np.sqrt(px_sig.sum())*self.params.inputs['coarse_grids']['peaking_time']
        num_of_wires_used = len(unique_x) + len(unique_y)
        snoise = self.ENC * (num_of_wires_used)**0.5
        print(f'Used {num_of_wires_used} wires, ({snoise:.2f} e)')
        if multiplier > 100:
            multiplier = 1

        return multiplier,err,num_of_wires_used


    def get_accurate_charge_from_wires(self):
        clusters = np.array(self.cluster_labels)
        total_charge = 0
        clustered_charges = []
        unique_clusters = np.unique(clusters[clusters>-1])
        for cluster_num in unique_clusters:
            val = self.sc_and_pws_from_cluster(cluster_num)
            # if there are no charges in this cluster, the val returned is False
            if val == False:
                print("\n***No charges detected on pixels, but wires triggered!***")
                # TODO add code to get charge from wires
                continue
            struck_chips, pixW_SIGNALS = val
            m,e,n = self.get_multiplier_for_cluster(struck_chips, pixW_SIGNALS)

            #charges in this cluster
            cmask = (clusters == cluster_num)
            r = self.total_charges_on_pixels[cmask].T
            q_on_px = r[-1].sum()
            total_charge += m * q_on_px

            DF = {}
            #DF['r'] = r
            DF['wire_q']  = m * q_on_px
            DF['pixel_q'] = q_on_px
            DF['multiplier'] = m

            clustered_charges.append(DF)
            print(f'Expected Error: {np.sqrt(n)*self.ENC*q_on_px/e:.2f}')

        self.total_charge_from_wires = total_charge
        self.clustered_charges = clustered_charges
        print(f'Total Charges    {total_charge/1000:4.2f}K e-')
        print(f'Deposited Charge {self.truth["num_electrons"]/1000:4.2f}K e- \n')
        print(f'Yield {total_charge/self.truth["num_electrons"]*100:4.2f}%')

        return clustered_charges



def find_bounding_box(r, buffer=0.0):
    """
    Finds box that spans r, with an added buffer
    """

    import numpy as np

    #   First set equal to extremes of r
    bounding_box = np.zeros((3,2))
    bounding_box[:, 0] = r.min(axis=1)
    bounding_box[:, 1] = r.max(axis=1)

    #   Add buffer
    bounding_box[:, 0] -= buffer
    bounding_box[:, 1] += buffer

    return bounding_box


def get_cell_bounds(track, charge_readout_name, readout_inputs_file_name,
                    compressed=True):
    """ Finds cell_bounds that contain raw track, buffering as
    needed to accomodate coarse sensors """

    from . import readout_tools
    from . import sims_tools

    #   Use compressed data if available and requested
    if compressed and hasattr(track, 'compressed'):
        r  = track.compressed['r']
    else:
        r  = track.raw['r']

    #   Find bounding dimensions that contains track
    cell_bounds = find_bounding_box(r)

    #   Buffer size to allow minimal coarse sensors
    coarse_pitch = readout_tools.Params(
        inputs_file_name = readout_inputs_file_name,
        charge_readout_name=charge_readout_name,
        ).coarse_pitch
    cell_bounds[:, 0] -= coarse_pitch
    cell_bounds[:, 1] += coarse_pitch

    return cell_bounds

# TRACKING CHANGES
# pixel sampling freq 4 -> 2 MHz
# changed the way tracks are called
# changed cutoff frequency from 1/peaking_time to np.pi/peaking_time

# below is the testing of the GAMPix algorithm
# it should be deleted and implemented from an outside file
