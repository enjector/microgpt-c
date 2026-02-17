/* MicroGPT-C — single-file C99 GPT. cc -O2 -o microgpt_amalgamated microgpt_amalgamated.c -lm
 * (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 * Decoder-only Transformer: Embed->RMSNorm->4-Head Attn->MLP(ReLU,4x)->lm_head
 * Trains on names.txt (one name per line), then generates new names. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
enum{E=16,H=4,D=E/H,B=16,F=E*4,ST=1000,WU=ST/10,BA=8,MV=257,MX=50000,NS=20};
#define LR 0.01
#define B1 0.85
#define B2 0.99
#define EP 1e-8
#define TP 0.5
#define Z(a,n) memset(a,0,(n)*8)
#define C(d,s,n) memcpy(d,s,(n)*8)
typedef struct{double*we,*wp,*lm,*wq,*wk,*wv,*wo,*f1,*f2;size_t vs;}G;
static unsigned long R=42;
static double ru(void){R=R*1103515245UL+12345UL;return(double)((R/65536UL)%32768UL)/32768.0;}
static double rg(void){double u=ru(),v=ru();if(u<1e-10)u=1e-10;return sqrt(-2*log(u))*cos(6.2831853*v);}
static int np(int v){return v*E*2+B*E+E*E*4+F*E+E*F;}
static double*ma(int n){double*p=malloc(n*8);for(int i=0;i<n;i++)p[i]=rg()*.08;return p;}
static G*gc(int v){G*g=calloc(1,sizeof(G));g->vs=v;g->we=ma(v*E);g->wp=ma(B*E);g->lm=ma(v*E);g->wq=ma(E*E);g->wk=ma(E*E);g->wv=ma(E*E);g->wo=ma(E*E);g->f1=ma(F*E);g->f2=ma(E*F);return g;}
static void gf(G*g){free(g->we);free(g->wp);free(g->lm);free(g->wq);free(g->wk);free(g->wv);free(g->wo);free(g->f1);free(g->f2);free(g);}
static void lf(const double*x,const double*W,int n,int o,double*y){for(int j=0;j<o;j++){double s=0;for(int i=0;i<n;i++)s+=x[i]*W[j*n+i];y[j]=s;}}
static void lb(const double*x,const double*W,const double*d,int n,int o,double*dx,double*dW){if(dx)for(int j=0;j<o;j++)for(int i=0;i<n;i++)dx[i]+=d[j]*W[j*n+i];if(dW)for(int j=0;j<o;j++)for(int i=0;i<n;i++)dW[j*n+i]+=d[j]*x[i];}
static void rn(const double*x,double*o){double s=0;for(int i=0;i<E;i++)s+=x[i]*x[i];double c=1/sqrt(s/E+1e-5);for(int i=0;i<E;i++)o[i]=x[i]*c;}
static void rb(const double*x,const double*dy,double*dx){double s=0,dt=0;for(int i=0;i<E;i++)s+=x[i]*x[i];double r=sqrt(s/E+1e-5),ir=1/r;for(int i=0;i<E;i++)dt+=dy[i]*x[i];double co=dt/(E*r*r);for(int i=0;i<E;i++)dx[i]+=ir*dy[i]-co*x[i];}
static void sf(double*a,int n){double m=a[0];for(int i=1;i<n;i++)if(a[i]>m)m=a[i];double s=0;for(int i=0;i<n;i++){a[i]=exp(a[i]-m);s+=a[i];}for(int i=0;i<n;i++)a[i]/=s;}
/* multi-head causal attention forward */
static void mha(const double*q,const double*k,const double*v,double*K,double*V,int*cl,double*aw,double*xa){int T=*cl+1;double sc=1/sqrt((double)D);C(K+*cl*E,k,E);C(V+*cl*E,v,E);for(int h=0;h<H;h++){int o=h*D,w=h*B;for(int t=0;t<T;t++){const double*kt=t<*cl?K+t*E+o:k+o;double s=0;for(int d=0;d<D;d++)s+=q[o+d]*kt[d];aw[w+t]=s*sc;}sf(aw+w,T);for(int d=0;d<D;d++){double s=0;for(int t=0;t<T;t++){const double*vt=t<*cl?V+t*E+o:v+o;s+=aw[w+t]*vt[d];}xa[o+d]=s;}}}
/* multi-head attention backward */
static void mhb(const double*da,const double*K,const double*V,const double*aw,const double*sq,int T,double*dq,double*dk,double*dv){double as=1/sqrt((double)D);for(int h=0;h<H;h++){int o=h*D,w=h*B;double dh[B],dt=0;for(int t=0;t<T;t++){double s=0;for(int d=0;d<D;d++)s+=da[o+d]*V[t*E+o+d];dh[t]=s;}for(int t=0;t<T;t++)dt+=aw[w+t]*dh[t];double ds[B];for(int t=0;t<T;t++)ds[t]=aw[w+t]*(dh[t]-dt)*as;for(int t=0;t<T;t++)for(int d=0;d<D;d++)dq[o+d]+=ds[t]*K[t*E+o+d];for(int d=0;d<D;d++){dk[o+d]+=ds[T-1]*sq[o+d];dv[o+d]+=aw[w+T-1]*da[o+d];}}}
/* fwd transformer block: attn->residual->MLP->residual (modifies x in-place, uses K/V cache) */
static void blk(const G*g,double*x,double*K,double*V,int*cl,double*aw,double*xa){double xn[E],q[E],k[E],v[E],x1[E],n2[E],h[F],x2[E];rn(x,xn);lf(xn,g->wq,E,E,q);lf(xn,g->wk,E,E,k);lf(xn,g->wv,E,E,v);mha(q,k,v,K,V,cl,aw,xa);lf(xa,g->wo,E,E,x1);for(int i=0;i<E;i++)x[i]=x1[i]+x[i];rn(x,n2);lf(n2,g->f1,E,F,h);for(int i=0;i<F;i++)h[i]=h[i]>0?h[i]:0;lf(h,g->f2,F,E,x2);for(int i=0;i<E;i++)x[i]=x2[i]+x[i];}
/* forward+backward one token position. returns cross-entropy loss */
static double fb(const G*g,int ti,int pi,int tg,double*K,double*V,int*cl,double*gb){int vs=(int)g->vs,T=*cl+1,ol=vs*E+B*E;double xe[E],x[E],xn[E],q[E],k[E],v[E],aw[H*B],xa[E],x1[E],n2[E],h[F],x2[E],lo[MV],Sxe[E],Sxp[E],Sxn[E],Sq[E],Saw[H*B],Sxa[E],Spa[E],Sn2[E],Shp[F],Shr[F];for(int i=0;i<E;i++)xe[i]=g->we[ti*E+i]+g->wp[pi*E+i];C(Sxe,xe,E);rn(xe,x);C(Sxp,x,E);rn(x,xn);C(Sxn,xn,E);lf(xn,g->wq,E,E,q);lf(xn,g->wk,E,E,k);lf(xn,g->wv,E,E,v);C(Sq,q,E);mha(q,k,v,K,V,cl,aw,xa);C(Saw,aw,H*B);C(Sxa,xa,E);lf(xa,g->wo,E,E,x1);for(int i=0;i<E;i++)x1[i]+=x[i];C(x,x1,E);C(Spa,x,E);rn(x,n2);C(Sn2,n2,E);lf(n2,g->f1,E,F,h);C(Shp,h,F);for(int i=0;i<F;i++)h[i]=h[i]>0?h[i]:0;C(Shr,h,F);lf(h,g->f2,F,E,x2);for(int i=0;i<E;i++)x2[i]+=x[i];C(x,x2,E);lf(x,g->lm,E,vs,lo);sf(lo,vs);double loss=-log(lo[tg]>1e-10?lo[tg]:1e-10);double dl[MV],dx[E],dm[F],dh[F],d2[E],dp[E],d1[E],da[E],dq[E],dk[E],dv[E],d0[E],de[E];for(int i=0;i<vs;i++)dl[i]=lo[i]-(i==tg?1:0);Z(dx,E);lb(x,g->lm,dl,E,vs,dx,gb+ol);Z(dm,F);lb(Shr,g->f2,dx,F,E,dm,gb+ol+vs*E+E*E*4+F*E);for(int i=0;i<F;i++)dh[i]=Shp[i]>0?dm[i]:0;Z(d2,E);lb(Sn2,g->f1,dh,E,F,d2,gb+ol+vs*E+E*E*4);Z(dp,E);rb(Spa,d2,dp);for(int i=0;i<E;i++)d1[i]=dx[i]+dp[i];Z(da,E);lb(Sxa,g->wo,d1,E,E,da,gb+ol+vs*E+E*E*3);Z(dq,E);Z(dk,E);Z(dv,E);mhb(da,K,V,Saw,Sq,T,dq,dk,dv);Z(d0,E);lb(Sxn,g->wq,dq,E,E,d0,gb+ol+vs*E);lb(Sxn,g->wk,dk,E,E,d0,gb+ol+vs*E+E*E);lb(Sxn,g->wv,dv,E,E,d0,gb+ol+vs*E+E*E*2);Z(dp,E);rb(Sxp,d0,dp);for(int i=0;i<E;i++)dp[i]+=d1[i];Z(de,E);rb(Sxe,dp,de);for(int i=0;i<E;i++){gb[ti*E+i]+=de[i];gb[vs*E+pi*E+i]+=de[i];}return loss;}
/* inference-only forward */
static void fi(const G*g,int ti,int pi,double*K,double*V,int*cl,double*lo){double x[E],xn[E],aw[H*B],xa[E];for(int i=0;i<E;i++)x[i]=g->we[ti*E+i]+g->wp[pi*E+i];rn(x,xn);C(x,xn,E);blk(g,x,K,V,cl,aw,xa);lf(x,g->lm,E,(int)g->vs,lo);++*cl;}
/* Adam optimiser with cosine LR + warmup */
static void ad(G*g,const double*gr,double*mo,double*vv,int s){double lr;if(s<WU)lr=LR*((double)(s+1)/WU);else{double p=(double)(s-WU)/(ST-WU);lr=LR*.5*(1+cos(p*3.14159265));}double c1=1-pow(B1,s+1.),c2=1-pow(B2,s+1.);int k=0,sz[]={(int)g->vs*E,B*E,(int)g->vs*E,E*E,E*E,E*E,E*E,F*E,E*F};double*aw[]={g->we,g->wp,g->lm,g->wq,g->wk,g->wv,g->wo,g->f1,g->f2};for(int p=0;p<9;p++)for(int i=0;i<sz[p];i++,k++){double gg=gr[k];mo[k]=B1*mo[k]+(1-B1)*gg;vv[k]=B2*vv[k]+(1-B2)*gg*gg;aw[p][i]-=lr*(mo[k]/c1)/(sqrt(vv[k]/c2)+EP);}}
/* temperature-scaled sampling */
static int sa(const double*lo,int vs){double b[MV],mx=lo[0];for(int i=1;i<vs;i++)if(lo[i]>mx)mx=lo[i];double s=0;for(int i=0;i<vs;i++){b[i]=exp((lo[i]-mx)/TP);s+=b[i];}for(int i=0;i<vs;i++)b[i]/=s;double r=ru();for(int i=0;i<vs;i++){r-=b[i];if(r<=0)return i;}return vs-1;}
int main(void){srand(42);R=42;FILE*f=fopen("names.txt","rb");if(!f){fprintf(stderr,"no names.txt\n");return 1;}fseek(f,0,SEEK_END);long fz=ftell(f);fseek(f,0,SEEK_SET);char*da=malloc(fz+1);fread(da,1,fz,f);da[fz]=0;fclose(f);
  char*ln[MX];int le[MX],nd=0;char*p=da;while(nd<MX&&*p){char*s=p;while(*p&&*p!='\r'&&*p!='\n')p++;int l=(int)(p-s);if(l){ln[nd]=s;le[nd]=l;nd++;}if(*p=='\r')p++;if(*p=='\n')p++;}
  for(int i=nd;i>1;i--){int j=rand()%i;char*t=ln[j];int ts=le[j];ln[j]=ln[i-1];le[j]=le[i-1];ln[i-1]=t;le[i-1]=ts;}
  unsigned char se[256]={0},ch[256];int nc=0;for(int i=0;i<nd;i++)for(int j=0;j<le[i];j++)se[(unsigned char)ln[i][j]]=1;for(int i=0;i<256;i++)if(se[i])ch[nc++]=i;
  int vs=nc+1,bo=nc;printf("docs:%d vocab:%d embd:%d heads:%d\n",nd,vs,E,H);G*g=gc(vs);int n=np(vs);printf("params:%d\n",n);
  double*gb=calloc(n,8),*mo=calloc(n,8),*vv=calloc(n,8),*K=malloc(B*E*8),*V=malloc(B*E*8);int cl,di=0;clock_t t0=clock();
  for(int st=0;st<ST;st++){Z(gb,n);double bl=0;int bp=0;for(int b=0;b<BA;b++){cl=0;char*dc=ln[di%nd];int dl=le[di%nd];di++;int tk[B+2],nt=0;tk[nt++]=bo;for(int i=0;i<dl&&nt<B+2;i++){int id=vs;for(int c=0;c<nc;c++)if(ch[c]==(unsigned char)dc[i]){id=c;break;}tk[nt++]=id;}if(nt<B+2)tk[nt++]=bo;int nn=nt-1;if(nn>B)nn=B;bp+=nn;for(int pos=0;pos<nn;pos++)bl+=fb(g,tk[pos],pos,tk[pos+1],K,V,&cl,gb);}for(int i=0;i<n;i++)gb[i]/=bp;ad(g,gb,mo,vv,st);if((st+1)%100==0||st==0)printf("step %4d/%d loss %.4f\n",st+1,ST,bl/bp);}
  printf("\nTrain:%.2fs\n--- names ---\n",(double)(clock()-t0)/CLOCKS_PER_SEC);double lo[MV];
  for(int s=0;s<NS;s++){cl=0;int ti=bo;char nm[B+1];int nl=0;for(int pos=0;pos<B;pos++){fi(g,ti,pos,K,V,&cl,lo);ti=sa(lo,vs);if(ti==bo)break;nm[nl++]=ch[ti];}nm[nl]=0;printf("  %2d: %s\n",s+1,nm);}
  free(gb);free(mo);free(vv);free(K);free(V);gf(g);free(da);return 0;}
