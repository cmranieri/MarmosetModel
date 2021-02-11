TITLE  All ion channels used in Str models



NEURON {
	SUFFIX Str
	NONSPECIFIC_CURRENT ilk,im,ik,ina, irand 
	RANGE gnabar, ena, alpha_m, alpha_h, beta_h, beta_m        : fast sodium
	RANGE gkdrbar, ek, alpha_n, beta_n                 		   : delayed K rectifier
	RANGE gl, el                                  	           : leak
	RANGE gmbar,em, alpha_p, beta_p							   : m current(outward potassium current)
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(S)  = (siemens)
	(molar) = (1/liter)
	(mM)	= (millimolar)
	FARADAY = (faraday) (coulomb)  :units are really coulombs/mole
}

PARAMETER {
	ena = 50	(mV)
	ek = -100	(mV)
	em = -100 	(mV)	

:Fast Na channel
	gnabar   = 100e-3 (S/cm2) 
	theta_m = 54 (mV)
	theta_h = 50 (mV) 
	alpha_m1=0.32	(ms/mV)
	th_ma=-54		(mV)
	k_m = 4 	(mV)    
	k_h = 18 	(mV)   
	beta_m1 = 0.28 (ms/mV)
	beta_h1 = 4 (ms) 
	alpha_h1=0.128	(ms/mV)

	th_mb=-27	(mV)
	tht_m = -27 (mV)
	tht_h2 = 27 (mV)
	sig_m = 5 (mV)
	sig_h2 = 5 (mV)

: delayed K rectifier 
	gkdrbar  = 80e-3	(S/cm2)  
	theta_n = 52 (mV)
	alpha_n1=0.032	(ms/mV)
	th_n=-52		(mV)
	k_n = 5 (mV)     
	beta_n1 = 0.5 (ms) 
	tht_n1 = 57 (mV)
	sig_n1 = 40 (mV)

:Leakage current
	gl	= 0.1e-3	(S/cm2)
	el	= -67	(mV)

:m current(outward potassium current)
	gmbar   = 2.6e-3 (S/cm2)  :Healthy=2.6 ,PD=1.5
	theta_p = 30 (mV)
	alpha_p1=3.209e-4	(ms/mV)
	th_p=-30	(mV)
	k_p = 9 (mV)    
	tht_p1 = -30 (mV)
	beta_p1=-3.209e-4	(ms/mV)
	thb_p=-30	(mV)
	sig_p1 = 9 (mV)
}

ASSIGNED {
	v	(mV)
	ina	(mA/cm2)
	irand (mA/cm2)
	ik	(mA/cm2)   
	ilk	(mA/cm2)
	im	(mA/cm2)

:Fast Na
	alpha_h	(1/ms)
	beta_h	(1/ms)
	alpha_m (1/ms)
	beta_m	(1/ms)

:K rectifier
	alpha_n	(1/ms)
	beta_n	(1/ms)

:m current(outward potassium current)
	alpha_p	(1/ms)
	beta_p	(1/ms)
}

STATE {
	m h n p 
	cai (mM) <1e-10> : Mas que droga eh essa aqui?
	cao (mM) <1e-10>
	nai (mM) <1e-10>
	nao (mM) <1e-10>
	ki (mM) <1e-10>
	ko (mM) <1e-10>
}

BREAKPOINT {
	SOLVE states METHOD cnexp

	ilk = gl * (v - el)					: leak
	ina = gnabar * m*m*m*h * (v - ena)	: fast sodium
	ik = gkdrbar * n^4 * (v - ek)		: delayed K rectifier
	im =  gmbar* p * (v - em)			: m current(outward potassium current)
	irand = 0*2e-3*sin(t/20)
}

DERIVATIVE states {   
	evaluate_fct(v)
	m' = alpha_m*(1-m)-beta_m*m
	h' = alpha_h*(1-h)-beta_h*h
	n' = alpha_n*(1-n)-beta_n*n
	p' = alpha_p*(1-p)-beta_p*p
}

UNITSOFF

INITIAL {
	evaluate_fct(v)
	m = alpha_m/(alpha_m+beta_m) 
	h = alpha_h/(alpha_h+beta_h) 
	n = alpha_n/(alpha_n+beta_n)  
	p = alpha_p/(alpha_p+beta_p) 
}

PROCEDURE evaluate_fct(v(mV)) { 
	
	alpha_m = alpha_m1*(v-th_ma)/(1-exp((-v-theta_m)/k_m))
	beta_m = beta_m1*(v-th_mb)/(-1+exp((v-tht_m)/sig_m)) 

	alpha_h = alpha_h1*exp((-v-theta_h)/k_h)
	beta_h = beta_h1/(1 + exp((-v-tht_h2)/sig_h2)) 

	alpha_n = alpha_n1*(v-th_n)/(1-exp((-v-theta_n)/k_n))
	beta_n = beta_n1*(exp((-v-tht_n1)/sig_n1))

	alpha_p = alpha_p1*(v-th_p)/(1-exp((-v-theta_p)/k_p))
	beta_p = beta_p1*(v-thb_p)/(1-exp((v-tht_p1)/sig_p1)) 
}

UNITSON
