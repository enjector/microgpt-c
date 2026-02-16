/* MicroGPT-C — single-file C99 GPT. Matches Andrej Karpathy's microgpt.py.
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 * Architecture: Embed -> RMSNorm -> Attention (4h, causal) -> MLP (ReLU, 4x) -> lm_head
 * Compile: cc -O2 -o microgpt microgpt_amalgamated.c -lm */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define NE 16
#define BS 16
#define HD (NE/4)
#define MD (NE*4)
#define STEPS 1000
#define LR 0.01
#define B1 0.85
#define B2 0.99
#define EP 1e-8
#define NS 20
#define TMP 0.5
#define MV 257
#define MX 50000
typedef struct { double *wte,*wpe,*lm,*wq,*wk,*wv,*wo,*f1,*f2; size_t vs; } G;
static unsigned long R=42;
static double ru(void) { R=R*1103515245UL+12345UL; return (double)((R/65536UL)%32768UL)/32768.0; }
static double rg(void) { double u=ru(),v=ru(); if(u<1e-10) u=1e-10; return sqrt(-2.*log(u))*cos(6.283185307*v); }
static size_t np(size_t v) { return v*NE*2+(size_t)BS*NE+NE*NE*4+MD*NE+NE*MD; }
static G *gc(size_t vs) {
  G *g=(G*)calloc(1,sizeof(G)); if(!g) return 0; g->vs=vs;
  size_t s[]={vs*NE,(size_t)BS*NE,vs*NE,NE*NE,NE*NE,NE*NE,NE*NE,MD*NE,NE*MD}; double **a=(double**)&g->wte;
  for(int i=0;i<9;i++) { a[i]=(double*)malloc(s[i]*8); if(!a[i]){free(g);return 0;} for(size_t j=0;j<s[i];j++) a[i][j]=rg()*.08; }
  return g; }
static void gf(G *g) { if(!g) return; double **a=(double**)&g->wte; for(int i=0;i<9;i++) free(a[i]); free(g); }
static void li(const double *x, const double *W, size_t n, size_t o, double *y) { for(size_t j=0;j<o;j++) { double s=0; for(size_t i=0;i<n;i++) s+=x[i]*W[j*n+i]; y[j]=s; } }
static void lb(const double *x, const double *W, const double *d, size_t n, size_t o, double *dx, double *dW) { if(dx) for(size_t i=0;i<n;i++) { double s=0; for(size_t j=0;j<o;j++) s+=d[j]*W[j*n+i]; dx[i]+=s; } if(dW) for(size_t j=0;j<o;j++) for(size_t i=0;i<n;i++) dW[j*n+i]+=d[j]*x[i]; }
static void rm(const double *x, size_t d, double *o) { double s=0; for(size_t i=0;i<d;i++) s+=x[i]*x[i]; double c=1./sqrt(s/(double)d+1e-5); for(size_t i=0;i<d;i++) o[i]=x[i]*c; }
static void sf(double *a, size_t n) { double m=a[0]; for(size_t i=1;i<n;i++) if(a[i]>m) m=a[i]; double s=0; for(size_t i=0;i<n;i++) { a[i]=exp(a[i]-m); s+=a[i]; } for(size_t i=0;i<n;i++) a[i]/=s; }
static void bk(const G *g, double *x, double **K, double **V, size_t *cl) {
  size_t T=cl[0]+1; double n[NE],q[NE],k[NE],v[NE],w[BS],a[NE],r[NE],n2[NE],h[MD],r2[NE];
  rm(x,NE,n); li(n,g->wq,NE,NE,q); li(n,g->wk,NE,NE,k); li(n,g->wv,NE,NE,v);
  memcpy(K[0]+cl[0]*NE,k,NE*8); memcpy(V[0]+cl[0]*NE,v,NE*8); double sc=1./sqrt((double)HD);
  for(size_t t=0;t<T;t++) { const double *kt=t<cl[0]?K[0]+t*NE:k; double s=0; for(size_t j=0;j<NE;j++) s+=n[j]*kt[j]; w[t]=s*sc; }
  sf(w,T); for(size_t j=0;j<NE;j++) { double s=0; for(size_t t=0;t<T;t++) s+=w[t]*(t<cl[0]?V[0][t*NE+j]:v[j]); a[j]=s; }
  li(a,g->wo,NE,NE,r); for(size_t i=0;i<NE;i++) x[i]=r[i]+x[i];
  rm(x,NE,n2); li(n2,g->f1,NE,MD,h); for(size_t i=0;i<MD;i++) h[i]=h[i]>0?h[i]:0;
  li(h,g->f2,MD,NE,r2); for(size_t i=0;i<NE;i++) x[i]=r2[i]+x[i]; }
static double fb(const G *g, size_t ti, size_t pi, size_t tg, double **K, double **V, size_t *cl, double *gb) {
  size_t vs=g->vs; double x[NE],xn[NE],lo[MV],dl[MV],dx[NE]; memset(dx,0,NE*8);
  for(size_t i=0;i<NE;i++) x[i]=g->wte[ti*NE+i]+g->wpe[pi*NE+i]; rm(x,NE,xn); memcpy(x,xn,NE*8);
  bk(g,x,K,V,cl); li(x,g->lm,NE,vs,lo); sf(lo,vs); double loss=-log(lo[tg]>1e-10?lo[tg]:1e-10);
  for(size_t i=0;i<vs;i++) dl[i]=lo[i]-(i==tg?1.:0.);
  lb(x,g->lm,dl,NE,vs,dx,gb+vs*NE+(size_t)BS*NE);
  for(size_t i=0;i<NE;i++) { gb[ti*NE+i]+=dx[i]; gb[vs*NE+pi*NE+i]+=dx[i]; }
  cl[0]++; return loss; }
static void fi(const G *g, size_t ti, size_t pi, double **K, double **V, size_t *cl, double *lo) {
  double x[NE],xn[NE]; for(size_t i=0;i<NE;i++) x[i]=g->wte[ti*NE+i]+g->wpe[pi*NE+i];
  rm(x,NE,xn); memcpy(x,xn,NE*8); bk(g,x,K,V,cl); li(x,g->lm,NE,g->vs,lo); cl[0]++; }
static void ad(G *g, const double *gr, double *m, double *v, int s) {
  double lr=LR*(1.-(double)s/STEPS), c1=1.-pow(B1,(double)(s+1)), c2=1.-pow(B2,(double)(s+1));
  size_t sz[]={g->vs*NE,(size_t)BS*NE,g->vs*NE,NE*NE,NE*NE,NE*NE,NE*NE,MD*NE,NE*MD};
  double **a=(double**)&g->wte; size_t k=0;
  for(int p=0;p<9;p++) for(size_t i=0;i<sz[p];i++,k++) { m[k]=B1*m[k]+(1-B1)*gr[k]; v[k]=B2*v[k]+(1-B2)*gr[k]*gr[k]; a[p][i]-=lr*(m[k]/c1)/(sqrt(v[k]/c2)+EP); } }
static size_t sa(const double *lo, size_t vs) {
  double b[MV],mx=lo[0]; for(size_t i=1;i<vs;i++) if(lo[i]>mx) mx=lo[i];
  double s=0; for(size_t i=0;i<vs;i++) { b[i]=exp((lo[i]-mx)/TMP); s+=b[i]; } for(size_t i=0;i<vs;i++) b[i]/=s;
  double r=ru(); for(size_t i=0;i<vs;i++) { r-=b[i]; if(r<=0) return i; } return vs-1; }
int main(void) {
  srand(42); R=42;
  FILE *f=fopen("names.txt","rb"); if(!f) { fprintf(stderr,"Cannot open names.txt\n"); return 1; }
  fseek(f,0,SEEK_END); long fz=ftell(f); fseek(f,0,SEEK_SET);
  char *da=(char*)malloc((size_t)fz+1); fread(da,1,(size_t)fz,f); da[fz]=0; fclose(f);
  char *ln[MX]; size_t le[MX],nd=0; char *p=da;
  while(nd<MX&&*p) { char *s=p; while(*p&&*p!='\r'&&*p!='\n') p++; size_t l=(size_t)(p-s); if(l) { ln[nd]=s; le[nd]=l; nd++; } if(*p=='\r') p++; if(*p=='\n') p++; }
  for(size_t i=nd;i>1;i--) { size_t j=(size_t)rand()%i; char *t=ln[j]; size_t ts=le[j]; ln[j]=ln[i-1]; le[j]=le[i-1]; ln[i-1]=t; le[i-1]=ts; }
  unsigned char se[256]={0},ch[256]; size_t nc=0;
  for(size_t i=0;i<nd;i++) for(size_t j=0;j<le[i];j++) se[(unsigned char)ln[i][j]]=1;
  for(int i=0;i<256;i++) if(se[i]) ch[nc++]=(unsigned char)i;
  size_t vs=nc+1,bo=nc; printf("docs: %zu  vocab: %zu\n",nd,vs);
  G *g=gc(vs); if(!g) { fprintf(stderr,"OOM\n"); return 1; } size_t n=np(vs); printf("params: %zu\n",n);
  double *gb=(double*)calloc(n,8), *mo=(double*)calloc(n,8), *vv=(double*)calloc(n,8);
  double *K=(double*)malloc((size_t)BS*NE*8), *V=(double*)malloc((size_t)BS*NE*8); size_t cl;
  clock_t t0=clock();
  for(int st=0;st<STEPS;st++) {
    memset(gb,0,n*8); cl=0; char *dc=ln[st%nd]; size_t dl=le[st%nd];
    size_t tk[BS+2],nt=0; tk[nt++]=bo;
    for(size_t i=0;i<dl&&nt<BS+2;i++) { size_t id=vs; for(size_t c=0;c<nc;c++) if(ch[c]==(unsigned char)dc[i]) { id=c; break; } tk[nt++]=id; }
    if(nt<BS+2) tk[nt++]=bo; size_t nn=nt-1; if(nn>BS) nn=BS;
    double tl=0; for(size_t pos=0;pos<nn;pos++) tl+=fb(g,tk[pos],pos,tk[pos+1],&K,&V,&cl,gb);
    for(size_t i=0;i<n;i++) gb[i]/=(double)nn; ad(g,gb,mo,vv,st);
    if((st+1)%100==0||st==0) printf("step %4d/%d  loss %.4f\n",st+1,STEPS,tl/(double)nn); }
  double ts=(double)(clock()-t0)/CLOCKS_PER_SEC;
  printf("\nTraining: %.2fs  %.0f steps/s\n",ts,(double)STEPS/ts);
  printf("\n--- generated names ---\n"); double lo[MV]; clock_t t1=clock();
  for(int s=0;s<NS;s++) { cl=0; size_t ti=bo; char nm[BS+1]; size_t nl=0;
    for(size_t pos=0;pos<BS;pos++) { fi(g,ti,pos,&K,&V,&cl,lo); ti=sa(lo,vs); if(ti==bo) break; nm[nl++]=(char)ch[ti]; }
    nm[nl]=0; printf("  %2d: %s\n",s+1,nm); }
  double is=(double)(clock()-t1)/CLOCKS_PER_SEC; if(is<.001) is=.001;
  printf("\nInference: %.3fs  %.0f samples/s\n",is,(double)NS/is);
  free(gb); free(mo); free(vv); free(K); free(V); gf(g); free(da); return 0; }


  