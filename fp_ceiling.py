# Потолок fp-only распознавания: реплика _nav_landmark_map/_nav_fingerprint из ячейки 30 (read-only анализ, код ноутбука не трогаем)
import numpy as np, math
W=208; R=4; SEED=8321; SM=32
m=np.random.RandomState(SEED).rand(W,W)
for _ in range(SM):
    m=(m+np.roll(m,1,0)+np.roll(m,-1,0)+np.roll(m,1,1)+np.roll(m,-1,1))/5.0
m=(m-m.min())/max(1e-9,float(m.max()-m.min()))
DIRS=((1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1))
px=[i for i in (-1,0,1) for _ in (-1,0,1)]; py=[j for _ in (-1,0,1) for j in (-1,0,1)]
ox,oy=list(px),list(py)
for dx,dy in DIRS:
    u=(dx/math.hypot(dx,dy),dy/math.hypot(dx,dy))
    cx0,cy0=int(round(u[0]*R)),int(round(u[1]*R))
    ox+=[cx0+i for i in px]; oy+=[cy0+j for j in py]
ox=np.array(ox); oy=np.array(oy)
# fp для всех клеток
X,Y=np.meshgrid(np.arange(W),np.arange(W),indexing='ij')
V=np.stack([m[(X+a)%W,(Y+b)%W] for a,b in zip(ox,oy)],axis=-1)   # W,W,81
fp=np.concatenate([V[...,:9].mean(-1,keepdims=True), V[...,9:].reshape(W,W,8,9).mean(-1)],axis=-1)  # W,W,9
print('fp shape',fp.shape,'карта: min/max',m.min(),m.max(),'std',m.std())
# 1) разброс fp по карте и градиент: насколько fp меняется за 1 клетку
d1=np.abs(fp-np.roll(fp,1,0)).mean(-1)
print('средн. |Δfp| за шаг в 1 клетку: mean=%.5f p50=%.5f p99=%.5f'%(d1.mean(),np.median(d1),np.percentile(d1,99)))
print('std fp по карте (по компонентам) mean=%.4f'%fp.std(axis=(0,1)).mean())
# 2) профиль различимости: mean|Δfp| как функция расстояния (по случайным парам на заданном расстоянии)
rng=np.random.RandomState(1)
def pair_stat(dist,n=4000):
    out=[]
    for _ in range(n):
        x,y=rng.randint(0,W,2); th=rng.rand()*2*math.pi
        x2=int(round(x+dist*math.cos(th)))%W; y2=int(round(y+dist*math.sin(th)))%W
        out.append(np.abs(fp[x,y]-fp[x2,y2]).mean())
    return np.array(out)
print('\nрасстояние -> mean|Δfp| (p50 / p10) ; доля пар с Δfp<=NOVEL_FP_TOL(0.045) | <=COARSE(0.12) | <=1.5*SIGMA_MIN(0.03)')
for d in (1,2,3,5,8,12,20,40,80):
    s=pair_stat(d)
    print('%3d  p50=%.4f p10=%.4f | %.2f | %.2f | %.2f'%(d,np.median(s),np.percentile(s,10),(s<=0.045).mean(),(s<=0.12).mean(),(s<=0.03).mean()))
# 3) алиасинг: для случайных клеток — сколько клеток В ДРУГОМ МЕСТЕ (>=15 клеток по тору) имеют fp в пределах tol
F=fp.reshape(-1,9)
idx=rng.choice(W*W,400,replace=False)
print('\nАлиасинг: доля клеток карты дальше 15 клеток, неотличимых по fp от случайной клетки')
def far_mask(i):
    x,y=divmod(i,W)
    dx=np.minimum(np.abs(X-x),W-np.abs(X-x)); dy=np.minimum(np.abs(Y-y),W-np.abs(Y-y))
    return (np.hypot(dx,dy)>=15).reshape(-1)
for tol in (0.03,0.045,0.12):
    fr=[];has=[]
    for i in idx:
        d=np.abs(F-F[i]).mean(-1)
        fm=far_mask(i)
        hit=(d<=tol)&fm
        fr.append(hit.sum()/fm.sum()); has.append(hit.any())
    fr=np.array(fr)
    print('tol=%.3f: средняя доля далёких клеток-двойников=%.4f (p90=%.4f) | клеток, у которых есть хоть один далёкий двойник=%.2f'%(tol,fr.mean(),np.percentile(fr,90),np.mean(has)))
# 4) "размер места" по fp: радиус, в пределах которого fp держится в tol (по случайным центрам), по 8 направлениям, макс. радиус до выхода
print('\nРадиус «места» по fp (до первого выхода за tol от fp центра), клеток:')
for tol in (0.03,0.045):
    rs=[]
    for i in idx[:200]:
        x,y=divmod(i,W); r_dirs=[]
        for dx,dy in DIRS:
            u=np.array([dx,dy])/math.hypot(dx,dy); r=0
            for k in range(1,60):
                xx=int(round(x+u[0]*k))%W; yy=int(round(y+u[1]*k))%W
                if np.abs(fp[xx,yy]-fp[x,y]).mean()>tol: break
                r=k
            r_dirs.append(r)
        rs.append(np.mean(r_dirs))
    rs=np.array(rs); print('tol=%.3f: средний радиус=%.1f  p10=%.1f p50=%.1f p90=%.1f'%(tol,rs.mean(),np.percentile(rs,10),np.median(rs),np.percentile(rs,90)))
