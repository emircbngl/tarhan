"""Kenar agirligi (cot a + cot b)/2 ne zaman negatif olur?
Belirleyici olan Delaunay kosulu mu, yoksa ucgenlerin dar acili olmasi mi?
Ortak kenar A-B; karsi koseler C (ustte) ve D (altta)."""
import math, random

def angle_at(p,q,r):
    v1=(q[0]-p[0],q[1]-p[1]); v2=(r[0]-p[0],r[1]-p[1])
    return math.atan2(abs(v1[0]*v2[1]-v1[1]*v2[0]), v1[0]*v2[0]+v1[1]*v2[1])
def cot(x): return math.cos(x)/math.sin(x)
def tri_obtuse(t):
    return max(angle_at(t[m],t[(m+1)%3],t[(m+2)%3]) for m in range(3)) > math.pi/2+1e-12

A,B=(0.0,0.0),(1.0,0.0)
print("  C ve D'yi AB'ye yaklastirdikca karsi acilar buyur (genis aci) :")
print(f"  {'h':>6} {'aci C':>8} {'aci D':>8} {'C+D':>8} {'agirlik':>10} {'Delaunay(C+D<=180)':>20}")
for h in (0.8,0.5,0.35,0.28,0.2,0.1):
    C=(0.5,h); D=(0.5,-h)
    gc=angle_at(C,A,B); gd=angle_at(D,A,B)
    w=0.5*(cot(gc)+cot(gd))
    print(f"  {h:6.2f} {math.degrees(gc):8.1f} {math.degrees(gd):8.1f} {math.degrees(gc+gd):8.1f} {w:10.4f} {str(gc+gd<=math.pi+1e-12):>20}")

print()
print("  KRITIK SORU: Delaunay saglanip da UCGEN genis acili olan durum var mi,")
print("  ve orada agirlik yine de pozitif mi? (1000 rastgele konfigurasyon)")
random.seed(0)
n_del=n_del_obt=n_del_neg=n_del_obt_pos=0
for _ in range(20000):
    C=(random.uniform(-1,2), random.uniform(0.05,1.5))
    D=(random.uniform(-1,2), random.uniform(-1.5,-0.05))
    gc=angle_at(C,A,B); gd=angle_at(D,A,B)
    delaunay = gc+gd <= math.pi+1e-12          # klasik Delaunay esdegeri
    if not delaunay: continue
    n_del+=1
    w=0.5*(cot(gc)+cot(gd))
    if w < -1e-12: n_del_neg+=1
    obt = tri_obtuse((A,B,C)) or tri_obtuse((A,B,D))
    if obt:
        n_del_obt+=1
        if w > 1e-12: n_del_obt_pos+=1
print(f"  Delaunay saglayan          : {n_del}")
print(f"  ...bunlarin NEGATIF agirlikli olani: {n_del_neg}")
print(f"  ...icinde genis acili ucgen olan   : {n_del_obt}")
print(f"  ......ve agirligi yine de POZITIF  : {n_del_obt_pos}")
