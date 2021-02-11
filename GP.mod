TITLE  All ion channels used in GP models



UNITSON

NEURON {
	SUFFIX GP
	NONSPECIFIC_CURRENT ilk, inaD, ikD, icaDT,icaD,iahp 
	RANGE gnabar, enaD, m_inf, h_inf, tau_h		         : fast sodium
	RANGE gkdrbar, ekD, n_inf, tau_n, ikDD                 : delayed K rectifier
	RANGE gl, el                                    : leak
	RANGE gcatbar, ecaD, s_inf 				 : T-type ca current
	RANGE r_inf	                         : ca dependent AHP K current
    RANGE a_inf    
    RANGE gahp
}


UNITS {
    (uA) = (microamp)
	(mA) = (milliamp)
	(mV) = (millivolt)
	(S)  = (siemens)
}

PARAMETER {
	enaD = 55	(mV)
	ekD = -80	(mV)
	ecaD = 120 	(mV)

:Fast Na channel
	gnabar   = 120e-3 (S/cm2) 
	theta_m = -37 (mV)
	theta_h = -58 (mV) 
	k_m = -10 (mV)    
	k_h = 12 (mV)   
	tau_h0 = 0.05 (ms)
	tau_h1 = 0.27 (ms) 
	tht_h2 = -40 (mV)
	sig_h2 = -12 (mV)

: delayed K rectifier 
	gkdrbar  = 30e-3	(S/cm2)  
	theta_n = -50 (mV)
	k_n = -14 (mV)     
	tau_n0 = 0.05 (ms)
	tau_n1 = 0.27 (ms) 
	tht_n2 = -40 (mV)
	sig_n2 = -12 (mV)

:Leakage current
	gl	= 0.1e-3	(S/cm2)
	el	= -65	(mV)

:T-type ca current
	gcatbar   = 0.15e-3 (S/cm2)  
	theta_s = -35 (mV)
	k_s = -2 (mV)    
	
:AHP current (Ca current)
	gt   = 0.5e-3 (S/cm2) 
	theta_r = -70 (mV)
	k_r = 2 (mV)
	tau_r = 30 (ms) 
	
	theta_a = -57 (mV)
	k_a = -2 (mV)
	
:AHP
	gahp = 10e-3 (S/cm2)

:cai
	k1_ca=1e-1 (cm2/mA/ms) 
	k2_ca=15e-3 (mA/cm2) 
	
}

ASSIGNED {
	v	(mV)
	inaD	(mA/cm2)
	ikD	(mA/cm2)   
	icaD	(mA/cm2)   
	icaDT(mA/cm2) 
	ilk	(mA/cm2)
	iahp (mA/cm2)

:Fast Na
	h_inf
	tau_h	(ms)
	m_inf

:K rectifier
	n_inf
	tau_n	(ms)

:T-type ca current
	s_inf

:AHP (Ca dependent K current)
	r_inf
	a_inf
	
}

STATE {
	h n r  
	CA	
}

BREAKPOINT {
	SOLVE states METHOD cnexp

	inaD = gnabar * m_inf*m_inf*m_inf*h * (v - enaD)
	ikD = gkdrbar * n^4 * (v - ekD)
	icaDT = gt *a_inf*a_inf*a_inf*r* (v - ecaD) 
	ilk = gl * (v - el)
	icaD = gcatbar * s_inf*s_inf * (v - ecaD)
	iahp=gahp*(v - ekD)*(CA/(CA+10))
}

DERIVATIVE states {   
	evaluate_fct(v)
	n' = 0.1*(n_inf - n)/tau_n
	h' = 0.05*(h_inf - h)/tau_h
	r' = (r_inf - r)/tau_r
	CA' =k1_ca*(-icaD-icaDT-k2_ca*CA)
}

UNITSOFF

INITIAL {
	evaluate_fct(v)
	h = h_inf 
	n = n_inf   
	r = r_inf 
	CA= 0.1 :pq = 0.1? n eh pra calcular a partir de icaD e icaDT?
}

PROCEDURE evaluate_fct(v(mV)) { 
	h_inf = 1/(1+exp((v-theta_h)/k_h))
	m_inf = 1/(1+exp((v-theta_m)/k_m))
	tau_h = tau_h0 + tau_h1/(1 + exp(-(v-tht_h2)/sig_h2)) 

	n_inf = 1/(1+exp((v-theta_n)/k_n))
	tau_n = tau_n0 + tau_n1/(1 + exp(-(v-tht_n2)/sig_n2))

	s_inf = 1/(1+exp((v-theta_s)/k_s))

	r_inf = 1/(1+exp((v-theta_r)/k_r))
	
	a_inf = 1/(1+exp((v-theta_a)/k_a))
}

UNITSON
