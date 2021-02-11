TITLE  All ion channels used in thalamic models



UNITSON

NEURON {
	SUFFIX thalamus

	NONSPECIFIC_CURRENT ilk,it,ik_th,ina_th 
	RANGE ina_th, ik_th
	RANGE gnabar, ena, m_inf, h_inf, tau_h		         : fast sodium
	RANGE gkdrbar, ek                 				 : delayed K rectifier
	RANGE gl, el, ilk                                    : leak
	RANGE eca, et, p_inf, icaT, gkcabar , r_inf			 : T-type ca current  
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(S)  = (siemens)
	(molar) = (1/liter)
}

PARAMETER {
	ena = 50	(mV)
	ek = -75	(mV)
	et = 0		(mV)

:Fast Na channel

	gnabar   = 3e-3 (S/cm2) 
	theta_m = -37 (mV)
	theta_h = -41 (mV) 
	k_m = -7 (mV)    
	k_h = 4 (mV)   

	tau_h0= 0.128 (1/ms)
	theta_h0= -46 (mV)
	k_h0= -18 (mV)
	tau_h1= 4 (1/ms)
	theta_h1 = -23 (mV)
	k_h1= -5 (mV)

: delayed K rectifier 

	gkdrbar  = 5e-3	(S/cm2)  

:Leakage current

	gl	= 0.05e-3	(S/cm2)
	el	= -70	(mV)

:Ca dynamics

:T-type ca current

	theta_p = -60 (mV)
	k_p = -6.2 (mV)    
	gt   = 5e-3 (S/cm2) 
	theta_r = -84 (mV)
	k_r = 4 (mV)
	th_r1=-25 (mV)
	tau_r1= -10.5 (mV)
}

ASSIGNED {
	v	(mV)
	ina_th	(mA/cm2)
	ik_th	(mA/cm2)     
	it	(mA/cm2) 
	ilk	(mA/cm2)

:Fast Na

	h_inf
	tau_h	(ms)
	m_inf

:T-type ca current

	p_inf

:AHP (Ca dependent K current)

	r_inf
	tau_r  (ms)
}

STATE {
	h r  
}

BREAKPOINT {
	SOLVE states METHOD cnexp

	ina_th = gnabar * m_inf*m_inf*m_inf*h * (v - ena)	: fast sodium
	ik_th = gkdrbar * (0.75*(1-h))^4 * (v - ek)		: delayed K rectifier
	ilk = gl * (v - el)								: leak
	it = gt * (v - et)*r*p_inf*p_inf				: T-type ca current
}

DERIVATIVE states {   
	evaluate_fct(v)

	h' = (h_inf - h)/tau_h
	r' = (r_inf - r)/tau_r
}

UNITSOFF

INITIAL {
	evaluate_fct(v)
	h = h_inf 
	r = r_inf 
}

PROCEDURE evaluate_fct(v(mV)) { 
	h_inf = 1/(1+exp((v-theta_h)/k_h))
	m_inf = 1/(1+exp((v-theta_m)/k_m))

	tau_h = 1/(tau_h0*exp((v-theta_h0)/k_h0)+tau_h1/(1+exp((v-theta_h1)/k_h1)))

	p_inf = 1/(1+exp((v-theta_p)/k_p))

	r_inf = 1/(1+exp((v-theta_r)/k_r))

	tau_r = 0.15*(28+exp((v-th_r1)/tau_r1))
}

UNITSON
