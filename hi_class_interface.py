import os
from cosmosis.datablock import names, option_section
import sys
import traceback
import pdb

#add class directory to the path
dirname = os.path.split(__file__)[0]
#enable debugging from the same directory
if not dirname.strip(): dirname='.'
install_dir = dirname+"/hi_class_public/classy_install/lib/python2.7/site-packages/"
sys.path.insert(0, install_dir)

import classy
import numpy as np

#These are pre-defined strings we use as datablock
#section names
cosmo = names.cosmological_parameters
distances = names.distances
cmb_cl = names.cmb_cl
growthparams  = names.growth_parameters 
horndeski = 'horndeski_parameters'

def setup(options):
    #Read options from the ini file which are fixed across
    #the length of the chain
    config = {
        'lmax': options.get_int(option_section,'lmax', default=2500),
        'zmax': options.get_double(option_section,'zmax', default=1.),
        'kmax': options.get_double(option_section,'kmax', default=1.0),
        'debug': options.get_bool(option_section, 'debug', default=False),
        'lensing': options.get_string(option_section, 'lensing', default = 'yes'),
        'expansion_model': options.get_string(option_section, 'expansion_model', default = 'lcdm'),
        'gravity_model':   options.get_string(option_section, 'gravity_model', default = 'propto_omega'),
        'modes':  options.get_string(option_section, 'modes', default = 's'),
        'output': options.get_string(option_section, 'output', default = 'tCl,lCl,pCl,mPk,mTk'),
        'sBBN file': options.get_string(option_section, 'sBBN_file'),
        #'skip_stability_tests_smg': options.get_string(option_section, 'skip_stability_tests_smg', default = 'no'),
        #'background_verbose': options.get_int(option_section,'background_verbose', default=1),
        #'thermodynamics_verbose': options.get_int(option_section,'thermodynamics_verbose', default=10)
        #'kineticity_safe_smg': options.get_double(option_section,'kineticity_safe_smg', default=1e-5)
            }
    #Create the object that connects to Class
    config['cosmo'] = classy.Class()

    #Return all this config information
    return config

def get_class_inputs(block, config):

    #Get parameters from block and give them the
    #names and form that class expects

    params = {
        'output':        config["output"],
        'modes':         config["modes"],
        'l_max_scalars': config["lmax"],
        'P_k_max_h/Mpc': config["kmax"],
        'lensing':       config["lensing"],
#        'background_verbose': config["background_verbose"],
        'z_pk': ', '.join(str(z) for z in np.arange(0.0, config['zmax'], 0.01)),
        'n_s':          block[cosmo, 'n_s'],
        'omega_b':      block[cosmo, 'ombh2'],
        'omega_cdm':    block[cosmo, 'omch2'],
        'tau_reio':     block[cosmo, 'tau'],
        'T_cmb':        block.get_double(cosmo, 't_cmb', default=2.726),
        'N_ur':         block.get_double(cosmo, 'N_ur', default=3.046),
        'k_pivot':      block.get_double(cosmo, 'k_pivot', default=0.05),
        }
    params['sBBN file'] = config['sBBN file']

    if block.has_value(cosmo, '100*theta_s'):
        params['100*theta_s'] = block[cosmo, '100*theta_s']
    if block.has_value(cosmo, 'h0'):
        params['H0'] = 100*block[cosmo, 'h0']
    if block.has_value(cosmo, 'A_s'):
        params['A_s'] = block[cosmo, 'A_s']
    if block.has_value(cosmo, 'logA'):
        params['ln10^{10}A_s'] = block[cosmo, 'logA']

    if block.has_value(horndeski, 'omega_fld'):
        print('Omega_fld not implemented at this stage due to conflict with cosmosis default\n \
               behaviour for Omega_Lambda')
        return 1
    
    if not block.has_value(horndeski, 'omega_smg') or (block[horndeski,'omega_smg'] == 0.):
        # standard CLASS case
        #params['Omega_fld'] = block.get_double(horndeski, 'omega_fld', default = 0.)
        print('Omega_smg = 0 or unspecified, running with default CLASS')
        #params['expansion_model'] = config['expansion_model']
    else:
        params['gravity_model'] =  config['gravity_model']
        params['expansion_model'] = config['expansion_model']
        smgs = smg_params(block)
        params['parameters_smg'] = block.get_string(horndeski, 'parameters_smg', default = smgs)
        params['kineticity_safe_smg'] = block.get_double(horndeski, 'kineticity_safe_smg', default=0.)

        if block.has_value(horndeski, 'omega_smg') and block[horndeski,'omega_smg'] < 0.:
            '''
            Omega_smg has a negqative value. In this case the equations for the scalar field
            will be used, you have to specify both Omega_Lambda and Omega_fld, and Omega_smg will
            be inferred by the code using the closure equation
            '''
            if config['expansion_model']=='lcdm':
                #params['expansion_smg'] = block.get_double(horndeski, 'omega_smg')
                params['expansion_smg'] = 0.5
            elif config['expansion_model']=='wowa':
                expansion_smg_string = '{0}, {1}, {2}'.format(0.5, block.get_double(cosmo, 'w'), block.get_double(cosmo, 'wa'))
                params['expansion_smg'] = expansion_smg_string

            params['Omega_Lambda'] = block.get_double(cosmo, 'omega_lambda', default = 0.)
            #params['Omega_fld'] = block.get_double(horndeski, 'omega_fld', default = 0.)
            params['Omega_fld'] = 0.0
            params['Omega_smg'] = block.get_int(horndeski, 'omega_smg', default = -1)

        elif block.has_value(horndeski, 'omega_smg') and (block[horndeski,'omega_smg'] > 0.) and (block[horndeski,'omega_smg'] < 1.):
            '''
            Omega_smg has a value larger than 0 but smaller than 1. In this case you should
            leave either Omega_Lambda or Omega_fld unspecified. Then, hi_class will run
            with the scalar field equations, and Omega_Lambda or Omega_fld will be inferred
            using the closure equation (sum_i Omega_i) equals (1 + Omega_k)
            '''
            if config['expansion_model']=='lcdm':
                params['expansion_smg'] = block.get_double(horndeski, 'omega_smg')
            elif config['expansion_model']=='wowa':
                expansion_smg_string = '{0}, {1}, {2}'.format(block.get_string(horndeski, 'omega_smg'), block.get_string(cosmo, 'w'), block.get_string(cosmo, 'wa'))

            params['Omega_smg'] = block.get_double(horndeski, 'omega_smg')

            '''
            if block.has_value(horndeski, 'omega_lambda_smg') and block.has_value(horndeski, 'omega_fld'):
                print('Both omega_lambda_smg and omega_fld specified, along with 0 < Omega_smg < 1\nOne should be left unspecfied')
                return 1 
            if block.has_value(horndeski, 'omega_lambda_smg'):
                params['Omega_Lambda'] = block.get_double(horndeski, 'omega_lambda_smg', default = 0.)
                # ToDo:
                # make sure omega-Lambda inferred from closure is used elsewhere.
            if block.has_value(horndeski, 'omega_fld'):
                params['Omega_fld'] = block.get_double(horndeski, 'omega_fld', default = 0.)
            '''

    if block.has_value(cosmo, 'N_ur') and block[cosmo,'N_ur'] != 3.046:
        params['N_ncdm'] = block[cosmo, 'N_ncdm']

        if block[cosmo, 'N_ncdm'] == 1:
            if block.has_value(cosmo, 'm_ncdm'):
                params['m_ncdm'] = block[cosmo, 'm_ncdm']
            if block.has_value(cosmo, 'omega_ncdm'):
                params['omega_ncdm'] = block[cosmo, 'omega_ncdm']
            params['T_ncdm'] = block[cosmo, 'T_ncdm']

        if block[cosmo, 'N_ncdm'] > 1:
            m_nu = []
            o_nu = []
            T_nu = []
            for i in range(1,4):
                if block.has_value(cosmo, 'm_ncdm__%i'%i):
                    m_nu.append(block[cosmo, 'm_ncdm__%i'%i])
                    T_nu.append(block[cosmo, 'T_ncdm__%i'%i])
                if block.has_value(cosmo, 'omega_ncdm__%i'%i):
                    o_nu.append(block[cosmo, 'omega_ncdm__%i'%i])
                    T_nu.append(block[cosmo, 'T_ncdm__%i'%i])
            print 'm_nu', len(m_nu)
            print 'omega_nu', len(o_nu)
            if len(m_nu)>0:
                params['m_ncdm'] = ",".join(map(str, m_nu))
                print 'm in'
            if len(o_nu)>0:
                print 'omega in'
                params['omega_ncdm'] = ",".join(map(str, o_nu))
            params['T_ncdm'] = ",".join(map(str, T_nu))

    return params

def get_class_outputs(block, c_classy, config):
    ##
    ## Derived cosmological parameters
    ##
    #pdb.set_trace()
    block[cosmo, 'sigma_8'] = c_classy.sigma8()
    h0 = c_classy.Hubble(0.)/100.#block[cosmo, 'h0']
    block[cosmo, 'omega_m'] = c_classy.Omega_m()
    block[cosmo, 'omega_lambda_smg'] = c_classy.Omega_smg()
    ##
    ##  Matter power spectrum
    ##

    #Ranges of the redshift and matter power
    dz = 0.01
    kmin = 1e-5 #1e-4
    kmax = config['kmax']*h0
    nk = 200 #1e-5

    #Define k,z we want to sample
    z = np.arange(0.0, config["zmax"]+dz, dz)
    k = np.logspace(np.log10(kmin), np.log10(kmax), nk)
    nz = len(z)

    #Extract (interpolate) P(k,z) at the requested
    #sample points.
    #P = np.zeros((nk,nz))
    P = np.zeros((nk, nz)) 
    for i,ki in enumerate(k):
        for j,zj in enumerate(z):
            #P[i,j] = c_classy.pk_lin(ki,zj)
            P[i,j] = c_classy.pk(ki,zj)

    #Save matter power as a grid
    block.put_grid("matter_power_lin", "k_h", k/h0, "z", z, "p_k", P*h0**3)
#    block.put_grid("matter_power_nl", "k_h", k/h0, "z", z, "p_k", P*h0**3)
    ##
    ##Distances and related quantities
    ##

    #save redshifts of samples
    block[distances, 'z'] = z
    block[distances, 'nz'] = nz

    #Save distance samples
    d_l = np.array([c_classy.luminosity_distance(zi) for zi in z])
    block[distances, 'd_l'] = d_l
    d_a = np.array([c_classy.angular_distance(zi) for zi in z])
    block[distances, 'd_a'] = d_a
    block[distances, 'd_m'] = d_a * (1+z)
    block[distances, 'H'] = np.array([c_classy.Hubble(zi) for zi in z])
    block[distances, 'mu'] = 5.*np.log10(d_l + 1e-100) + 25.
    
    #Save some auxiliary related parameters
    block[distances, 'age'] = c_classy.age()
    block[distances, 'rs_zdrag'] = c_classy.rs_drag()
    block[distances, 'a'] = 1./(1.+z)

    ## Growth stuff
    s8 = np.array([c_classy.sigma8_at_z(zi) for zi in z])
    grr = np.array([c_classy.growthrate_at_z(zi) for zi in z])
    
    # Save growth stuff
    #block[names.growth, 'sigma8_at_z'] = s8
    #block[names.growth, 'growthrate_at_z'] = grr
    block[growthparams, 'z'] = z
    f_z = np.array([c_classy.linear_growth_rate(zi) for zi in z])
    block[growthparams, 'f_z'] = f_z
    D_z = np.array([c_classy.linear_growth_factor(zi) for zi in z])
    block[growthparams, 'D_z'] = D_z


    ##
    ## Now the CMB C_ell
    ##
    if config['lensing'] == 'no':
        c_ell_data =  c_classy.raw_cl()
    if config['lensing'] == 'yes':
        c_ell_data = c_classy.lensed_cl()
    ell = c_ell_data['ell']
    ell = ell[2:]

    #Save the ell range
    block[cmb_cl, "ell"] = ell

    #t_cmb is in K, convert to mu_K, and add ell(ell+1) factor
    tcmb_muk = block[cosmo, 't_cmb'] * 1e6
    f = ell*(ell+1.0) / 2 / np.pi * tcmb_muk**2
    f1 = ell*(ell+1.0) / 2 / np.pi

    #Save each of the four spectra
    for s in ['tt','ee','te','bb','tp']:
        block[cmb_cl, s] = c_ell_data[s][2:] * f
    block[cmb_cl, 'pp'] = c_ell_data['pp'][2:] * f1

def execute(block, config):
    c_classy = config['cosmo']

   # try:
        # Set input parameters
    params = get_class_inputs(block, config)
    c_classy.set(params)
    #print(c_classy.pars)
    try:
        # Run calculations
        c_classy.compute()
        # Extract outputs
        get_class_outputs(block, c_classy, config)
    except classy.CosmoError as error:
        if config['debug']:
            sys.stderr.write("Error in class. You set debug=T so here is more debug info:\n")
            traceback.print_exc(file=sys.stderr)
        else:
            sys.stderr.write("Error in class. Set debug=T for info: {}\n".format(error))
        return 1
    finally:
        #Reset for re-use next time
        c_classy.struct_cleanup()
    return 0

def cleanup(config):
    config['cosmo'].empty()

def smg_params(block):
    snl =[]
    for i in range(1,20):
        if block.has_value(horndeski, 'parameters_smg__%i'% i):
            snl.append(block[horndeski, 'parameters_smg__%i'% i]) # "propto_omega" -> x_k, x_b, x_m, x_t, M*^2_ini (default)
        else:
            break
    smg = ",".join(map(str, snl))
    return smg

def smg_exp(block):
    snl_exp =[]
    for i in range(1,20):
        if block.has_value(cosmo, 'expansion_smg__%i'% i):
            snl_exp.append(block[cosmo, 'expansion_smg__%i'% i])
        else:
            break
    smg_exp = ",".join(map(str, snl_exp))
    return smg_exp
