TITLE  STN ion channels for single compartment model

COMMENT
//program from Basal ganglia network model of subthalamic deep brain stimulation (Hahn and McIntyre 2010)
//derived from: Otsuka model
//available on : 
//https://senselab.med.yale.edu/modeldb/ShowModel.cshtml?model=127388
//Author: Daniel Tamashiro, Unicamp 
//Updated 08/16/2016
ENDCOMMENT

UNITSON

NEURON {
	SUFFIX STN
	NONSPECIFIC_CURRENT ilk
	NONSPECIFIC_CURRENT icaT,icaL,ikD,ikA,ikAHP,ina, idbs

	RANGE gnabar, ena, m_inf, h_inf, tau_h, tau_m		   : fast sodium
	RANGE gkdrbar, ek, n_inf, tau_n                   : delayed K rectifier
	RANGE gl, el                                     : leak
	RANGE gcatbar, p_inf, tau_p, q_inf, tau_q	       : T-type ca current
	RANGE gcalbar, eca, c_inf, d1_inf, d2_inf, tau_c, tau_d1, tau_d2  : L-type ca current
	RANGE gkabar, ek, a_inf, tau_a, b_inf, tau_b     : A-type K current
	RANGE gkcabar, ek, r_inf                       : ca dependent AHP K current
	
	RANGE periodo, tezao, cai, dbs
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(S)  = (siemens)
	(molar) = (1/liter)
	(mM)	= (millimolar)
}

PARAMETER {
	ena = 60	(mV)
	ek = -90	(mV)
	cao = 2000	
	con =12.8392 (mV)
	
	tezao = 7.6923 (ms)
	dbs

:Fast Na channel
	gnabar   = 49e-3 (S/cm2) 
	theta_m = -40 (mV)
	theta_h = -45.5 (mV) 
	k_m = -8 (mV)   
	k_h = 6.4 (mV)   
	tau_m0 = 0.2 (ms)
	tau_m1 = 3 (ms)
	tau_h0 = 0 (ms)
	tau_h1 = 24.5 (ms) 
	tht_m = -53 (mV)
	tht_h1 = -50 (mV)
	tht_h2 = -50 (mV)
	sig_m = -0.7 (mV)
	sig_h1 = -15 (mV)
	sig_h2 = 16 (mV)

: Delayed rectifier K
	gkdrbar  = 57e-3	(S/cm2)  
	theta_n = -41 (mV)
	k_n = -14 (mV)     
	tau_n0 = 0 (ms)
	tau_n1 = 11 (ms) 
	tht_n1 = -40 (mV)
	tht_n2 = -40 (mV)
	sig_n1 = -40 (mV)
	sig_n2 = 50 (mV) 

:Leakage current
	gl	= 0.35e-3	(S/cm2)
	el	= -60	(mV)

:T-type ca current
	gcatbar   = 5e-3 (S/cm2)  
	theta_p = -56 (mV)
	theta_q = -85 (mV) 
	k_p = -6.7 (mV)    
	k_q = 5.8 (mV)  
	tau_p0 = 5 (ms)
	tau_p1 = 0.33 (ms)
	tau_q0 = 0 (ms)
	tau_q1 = 400 (ms) 
	tht_p1 = -27 (mV)
	tht_p2 = -102 (mV)
	tht_q1 = -50 (mV)
	tht_q2 = -50 (mV)
	sig_p1 = -10 (mV)
	sig_p2 = 15 (mV)	
	sig_q1 = -15 (mV)
	sig_q2 = 16 (mV)	

:Ca L current
	gcalbar   = 15e-3 (S/cm2) 
	theta_c = -30.6 (mV)
	theta_d1 = -60 (mV)
	theta_d2 = 0.1 (mV)
	k_c = -5 (mV) 	
	k_d1 = 7.5 (mV)
	k_d2 = 0.02 (mV)
	tau_c0 = 45 (ms)
	tau_c1 = 10 (ms)
	tau_d10 = 400 (ms)
	tau_d11 = 500 (ms)
	tht_c1 = -27 (mV)
	tht_c2 = -50 (mV)
	tht_d11 = -40 (mV)
	tht_d12 = -20 (mV)
	sig_c1 = -20 (mV)
	sig_c2 = 15 (mV)	
	sig_d11 = -15 (mV)
	sig_d12 = 20 (mV)	
	tau_d2 = 130 (ms)

:A current
	gkabar  = 5e-3	(S/cm2)  
	theta_a = -45 (mV)	
	theta_b = -90 (mV) 
	k_a = -14.7 (mV)    
	k_b = 7.5 (mV)   	
	tau_a0 = 1 (ms)
	tau_a1 = 1 (ms)
	tau_b0 = 0 (ms)
	tau_b1 = 200 (ms) 
	tht_a = -40 (mV)
	tht_b1 = -60 (mV)
	tht_b2 = -40 (mV)
	sig_a = -0.5 (mV)
	sig_b1 = -30 (mV)
	sig_b2 = 10 (mV)	

:AHP current (Ca dependent K current)
	gkcabar   = 1e-3 (S/cm2) 
	theta_r = 0.17 (mV) 
	k_r = -0.08 (mV) 
	tau_r = 2 (ms)

:cai 
	acai=5.18e-3 (cm2/mA/ms)
	bcai=2e-3 (1/ms) 
}

ASSIGNED {
	v	(mV)
	ina	(mA/cm2)
	ikD	(mA/cm2)   
	ikA	(mA/cm2) 
	ikAHP(mA/cm2)  
	icaT(mA/cm2) 
	icaL (mA/cm2)
	ilk	(mA/cm2)
	idbs (mA/cm2)
	
	periodo (ms)

:Fast Na
	h_inf
	tau_h	(ms)
	m_inf
	tau_m	(ms) 

:Delayed rectifier
	n_inf
	tau_n	(ms)

:ca T current
	p_inf
	q_inf
	tau_p	(ms)
	tau_q	(ms)
	eca     (mV)   :calc from Nernst

:ca L current
	c_inf
	tau_c	(ms)
	d1_inf
	tau_d1	(ms)
	d2_inf

:A current
	a_inf
	tau_a	(ms)
	b_inf
	tau_b	(ms)

:AHP (Ca dependent K current)
	r_inf
}

STATE {
	m h n
	p q 
	c d1 d2
	cai 
	a b r
}


BREAKPOINT {
	SOLVE states METHOD cnexp

	eca = con*log(cao/cai)

	ina   = gnabar * m*m*m*h * (v - ena)
	ikD   = gkdrbar * n^4 * (v - ek)
	
	ikA   = gkabar * a*a*b * (v - ek)
	ikAHP   = gkcabar *r*r* (v - ek)
	
	icaT   = gcatbar * p*p*q * (v - eca)
	icaL   = gcalbar * c*c*d1*d2 * (v - eca)

	ilk = gl * (v - el)
	
	if (t >= periodo + tezao){
		periodo = periodo + tezao
	}
	if (t >= periodo && t <= periodo + 0.3) { 
        idbs = -0.3*dbs
    } else{
		idbs = 0
    }
}

DERIVATIVE states {   
	evaluate_fct(v)
	h' = (h_inf - h)/tau_h
	m' = (m_inf - m)/tau_m
	n' = (n_inf - n)/tau_n
	p' = (p_inf - p)/tau_p
	q' = (q_inf - q)/tau_q

	c' = (c_inf - c)/tau_c
	d1' = (d1_inf - d1)/tau_d1
	d2' = (d2_inf - d2)/tau_d2

	cai' =	-acai*(icaL+icaT)-bcai*cai

	a' = (a_inf - a)/tau_a
	b' = (b_inf - b)/tau_b

	r' = (r_inf - r)/tau_r
}

UNITSOFF

INITIAL {
	evaluate_fct(v)
	m = m_inf 
	h = h_inf   
	n = n_inf   
	p = p_inf 
	q = q_inf  

	c = c_inf 
	d1 = d1_inf  
	d2 = d2_inf   

	a = a_inf 
	b = b_inf   

	r = r_inf 
	cai= 0.005
	
	periodo = 0
}

PROCEDURE evaluate_fct(v(mV)) { 
:Fast Na current
	h_inf = 1/(1+exp((v-theta_h)/k_h))
	m_inf = 1/(1+exp((v-theta_m)/k_m))
	tau_h = tau_h0 + tau_h1/(exp(-(v-tht_h1)/sig_h1) + exp(-(v-tht_h2)/sig_h2)) 
	tau_m = tau_m0 + tau_m1/(1+exp(-(v-tht_m)/sig_m)) 

:Delayed rectifier K
	n_inf = 1/(1+exp((v-theta_n)/k_n))
	tau_n = tau_n0 + tau_n1/(exp(-(v-tht_n1)/sig_n1) + exp(-(v-tht_n2)/sig_n2)) 

:Ca T current
	p_inf = 1/(1+exp((v-theta_p)/k_p))
	q_inf = 1/(1+exp((v-theta_q)/k_q))
	tau_p = tau_p0 + tau_p1/(exp(-(v-tht_p1)/sig_p1) + exp(-(v-tht_p2)/sig_p2)) 
	tau_q = tau_q0 + tau_q1/(exp(-(v-tht_q1)/sig_q1) + exp(-(v-tht_q2)/sig_q2))

:Ca L current
	c_inf = 1/(1+exp((v-theta_c)/k_c))
	d1_inf = 1/(1+exp((v-theta_d1)/k_d1))
	d2_inf = 1/(1+exp((v-theta_d2)/k_d2))
	tau_c = tau_c0 + tau_c1/(exp(-(v-tht_c1)/sig_c1) + exp(-(v-tht_c2)/sig_c2))  
	tau_d1 = tau_d10 + tau_d11/(exp(-(v-tht_d11)/sig_d11) + exp(-(v-tht_d12)/sig_d12))  
	
:A current
	a_inf = 1/(1+exp((v-theta_a)/k_a))
	b_inf = 1/(1+exp((v-theta_b)/k_b))
	tau_a = tau_a0 + tau_a1/(1+exp(-(v-tht_a)/sig_a))
	tau_b = tau_b0 + tau_b1/(exp(-(v-tht_b1)/sig_b1) + exp(-(v-tht_b2)/sig_b2))  
	
:AHP current
	r_inf = 1/(1+exp((v-theta_r)/k_r))
}

UNITSON