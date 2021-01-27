NEURON {
    POINT_PROCESS Izhi2003b : qual eh a dif de fazer um processo pontual ou distribuido?
    RANGE a,b,c,d,f,g,thresh
    NONSPECIFIC_CURRENT i
}

UNITS {
    (mV) = (millivolt)
    (nA) = (nanoamp)
    (nF) = (nanofarad)
}

INITIAL {
  v=-65 
  u=0.2*v
  net_send(0,1) : oq eh isso?
}

PARAMETER {
    a       = 0.02 (/ms)
    b       = 0.2  (/ms)
    c       = -65  (mV)   : reset potential after a spike
    d       = 2    (mV/ms) 
    f = 5
    g = 140
    thresh = 30   (mV)   : spike threshold
}

ASSIGNED {
    v (mV)
    i (nA)
}

STATE {
    u (mV/ms)
}

BREAKPOINT {
    SOLVE states METHOD derivimplicit  : cnexp # either method works
    i = -0.001*(0.04*v*v + f*v + g - u ) 
}

DERIVATIVE states {
    u' = a*(b*v - u)
}

NET_RECEIVE (w) { : n entendi
    if (flag == 1) { 
        WATCH (v > thresh) 2
    } else if (flag == 2) {
        net_event(t)
        v = c
        u = u + d
    }
}