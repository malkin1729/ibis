import torch as T
import numpy as np
import gc

def score(model, padded):
    
    outputs = model(padded)
    
    lsm = -outputs[0].log_softmax(2)
    preds = T.zeros_like(lsm)
    preds[:,1:] = lsm[:,:-1]
    wordscores = preds.gather(2,padded.unsqueeze(2)).squeeze(2)
    scores = wordscores.sum(1)
    
    return scores.cpu(), wordscores.cpu(), -preds.cpu()

cand_orders = { 3: [[1,3,2,4]], 
                4: [[1,4,3,2,5]], 
                5: [[1,3,2,5,4,6],[1,3,5,2,4,6],[1,3,5,4,2,6],[1,4,2,5,3,6],[1,4,3,5,2,6],[1,5,4,3,2,6],[1,5,2,4,3,6],[1,5,3,2,4,6]] }

def shuffle_proposals(mat, topk, bs, kopt):
        
    L = mat.shape[0]

    I = T.zeros((kopt,)+(L,)*(kopt)).long()
    for i in range(kopt):
        I[i] = T.arange(L).view((-1,) + (1,)*(kopt-1-i))
    
    mask = (0<I[0]) 
    for i in range(kopt-1):
        mask *= (I[i]<I[i+1])
    
    lv = mat.view(-1)
   
    orders = cand_orders[kopt]
        
    o = np.array(orders[np.random.randint(len(orders))])
    then = T.zeros((L,)*kopt)
    now = T.zeros_like(then)
    for i in range(kopt):
        now += lv[ L*I[i] + I[i] ]
        then += lv[ L*I[o[i]-1] + I[o[i+1]-2] ]
        
    A = then - now

    A[~mask] = -1001

    topv, topi = A.view(-1).topk(min(A.numel(), topk))
    indices = np.random.randint(topi.shape[0],size=(bs,))
    topv = topv[indices]
    topi = topi[indices]
    
    orders = [o] * bs
        
    imod = [(topi//L**(kopt-1-i))%L for i in range(kopt)]
    
    return T.stack(imod,-1), topv, orders

def ibis(model, device, before, sentence, after, bs, topk, its, patience, warminit=False, gluemask=None):
    
    sent = sentence
    
    padded = T.cat([before,sent,after],0).unsqueeze(0).to(device)
    
    zz = score(model, padded)
    orscore = zz[0][0]
    yield orscore

    bestscore = zz[0][0]
    
    bestsc = zz[2][0]
    
    lfix,rfix,blanks=before.shape[0]-1,after.shape[0]-1,0
    
    permsents = [ T.cat([before, T.from_numpy((sent.numpy())), after],0) for _ in range(bs) ]

    bestmask = np.full(permsents[0].shape, True)
    
    if gluemask is not None: bestmask[lfix+1:-rfix-1] = gluemask
    
    permmasks = [ bestmask.copy() for _ in range(bs) ]
    
    if not warminit:
        seg = list(np.nonzero(bestmask[lfix+1:-rfix-1])[0]) + [ len(sent) ]
        for b in range(bs):
            perm = np.random.permutation(len(seg)-1)
            ns = []
            nm = []
            for i in range(len(seg)-1):
                ns.append(sent[seg[perm[i]]:seg[perm[i]+1]])
                nm.append(bestmask[lfix+1:-rfix-1][seg[perm[i]]:seg[perm[i]+1]])
            permsents[b][lfix+1:-rfix-1] = T.cat( ns, 0 )
            permmasks[b][lfix+1:-rfix-1] = np.concatenate( nm, 0 )
        
    padded = T.stack(permsents,0).to(device)
    
    bestsent = np.zeros(padded[0].shape)
       
    bestscore = 1000000
    movetype = 'init'
    nch = 0
    candidates = np.array([1]*bs)
    last_imp = 0
    for it in range(its):
        
        gc.collect()
        
        if it-last_imp > patience: break

        sc,wsc,spr = score(model, padded)

        if it == 0: bestwsc=wsc[0]
            
        sc = sc.numpy()
        
        if sc.min() < bestscore:
            if it == 0 or np.any(permsents[sc.argmin()] != bestsent):

                nch += 1
  
                bestsent,bestscore,bestsc,bestwsc, bestmask = permsents[sc.argmin()], sc.min(), spr[sc.argmin()], wsc[sc.argmin()], permmasks[sc.argmin()]
    
                if type(bestsent)==T.Tensor: bestsent = bestsent.numpy()
                
                last_imp = it
                
                yield (it, movetype, bestscore, bestsent, bestmask)
               
        thespr = bestsc

        kopt = np.random.randint(3,6)

        cutprobs = np.ones_like(bestwsc)

        cutprobs[~bestmask] = 0.
        
        cutprobs[lfix] = 100
        cutprobs[-1-rfix] = 100
        
        if it%2 == 0 and len(bestsent)-lfix-rfix > 6:
            ncand = bestmask[lfix:len(bestsent)-rfix].sum()
            
            if kopt == 4: ncand = min(40,ncand)
            if kopt == 5: ncand = min(20,ncand)
            
            l,r = lfix, len(bestsent)-rfix
            candidates = np.random.choice(np.arange(l,r), replace=False, p=cutprobs[l:r]/cutprobs[l:r].sum(), size=(ncand,))
            candidates.sort()

            movetype=f'GS {kopt}'

        else:
            
            ropt = np.random.randint(7,15)
            
            try:
                start = np.random.randint(lfix+1, len(bestsent)-ropt-rfix)

                l,r = start,start+ropt
            
                candidates = np.random.choice(np.arange(l,r), replace=False, p=cutprobs[l:r]/cutprobs[l:r].sum(), size=(min(ropt,(cutprobs[l:r]>0).sum()),))
                
            except:
                ropt = min(15,len(bestsent)-lfix-rfix-2)
                start = np.random.randint(lfix+1,max(lfix+2,len(bestsent)-ropt-rfix))
            
                l,r = start,start+ropt
                candidates = np.random.choice(np.arange(l,r), replace=False, p=cutprobs[l:r]/cutprobs[l:r].sum(), size=(min(ropt,(cutprobs[l:r]>0).sum()),))
            
            candidates.sort()

            movetype=f'LS {kopt}'
        
        links = thespr[:,bestsent[candidates]][candidates]

        permsents = []
        permmasks = []
     
        i,v,o = shuffle_proposals(links, topk, bs, kopt)
        
        for j in range(bs):
            inds = [ candidates[0] ] + list(candidates[i[j]]) + [ candidates[-1] ]
            if v[j] > -1000:
                pieces = [bestsent[:inds[0]]]
                maskpieces = [bestmask[:inds[0]]]
                for k in range(kopt+1):
                    pieces.append(bestsent[inds[o[j][k]-1]:inds[o[j][k]]])
                    maskpieces.append(bestmask[inds[o[j][k]-1]:inds[o[j][k]]])
                pieces.append(bestsent[inds[-1]:])
                newsent = np.concatenate(pieces,0)
                maskpieces.append(bestmask[inds[-1]:])
                newmask = np.concatenate(maskpieces,0)
            else: newsent, newmask = bestsent, bestmask
                
            permsents.append(newsent)
            permmasks.append(newmask)
            
        padded = T.stack(list(map(T.from_numpy,permsents)),0).to(device)