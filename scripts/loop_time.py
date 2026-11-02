# Inference-loop seconds from koine_machines INFO timestamps: 'Selected N patches' -> 'Wrote'
import re,sys,datetime as dt
for R in sys.argv[1:]:
    t=open(R+'/time.log').read(); ts=lambda l: dt.datetime.strptime(l[1:24],'%Y-%m-%d %H:%M:%S,%f')
    L=t.splitlines(); first=[l for l in L if l.startswith('[20')]
    sel=[l for l in L if 'Selected' in l and 'patches' in l]; wr=[l for l in L if 'INFO Wrote' in l]
    wall=re.search(r'Elapsed \(wall clock\).*?: (\S+)',t).group(1)
    if sel and wr:
        n=int(re.search(r'Selected (\d+) patches',sel[0]).group(1)); s=(ts(wr[-1])-ts(sel[0])).total_seconds()
        print(R.split('/')[-1],'patches',n,'loop_s',round(s,2),'startup_s',round((ts(sel[0])-ts(first[0])).total_seconds(),2),'wall',wall,'cm2_per_min_loop',round(7.2/s*60,1))
    else: print(R.split('/')[-1],'no loop timing; wall',wall)
