from __future__ import annotations

import argparse, csv, statistics, threading, time
from dataclasses import dataclass
import torch

from specdecode.models import load_model_and_tokenizer
from specdecode.policy import make_acceptance_history_policy, make_combined_policy, make_entropy_only_policy, make_latency_aware_policy
from specdecode.reward import ThroughputRateReward
from specdecode.speculative import SpeculativeDecoder
from specdecode.harness import RuntimeRegime, run_under_regime

FACTORIES={"entropy":make_entropy_only_policy,"acceptance":make_acceptance_history_policy,"latency":make_latency_aware_policy,"combined":make_combined_policy}
FIXED={"fixed_1":1,"fixed_3":3,"fixed_5":5,"fixed_8":8}
VARIANTS=list(FIXED)+list(FACTORIES)
TRAIN_PROMPTS=["Artificial intelligence systems are increasingly used to","A programmer debugging a difficult problem should","The development of modern processors has enabled","Researchers studying language models found that","In the early days of the internet,","A reliable distributed system must handle","The spacecraft transmitted new data about","One important challenge in machine learning is","The city began changing rapidly after","When learning a complicated technical subject,"]
TEST_PROMPTS=["Advances in computer hardware have made it possible to","A large online platform must respond quickly when","The professor explained the difficult concept by","During the mission the control system detected","The software team improved the application after","A new generation of language models may","When many requests arrive at the same time,","The experiment was repeated under different conditions because","Computer scientists often evaluate a system by measuring","After the unexpected change in workload the system"]

def train_policy(decoder,policy,tokens):
    for i,users in enumerate([1,2,4,6,8]):
        prompts=[TRAIN_PROMPTS[(i+j)%len(TRAIN_PROMPTS)] for j in range(users)]
        run_under_regime(decoder,prompts,tokens,policy,RuntimeRegime(f"train_{users}",users,0),do_sample=False,record_runtime_stats=False)
    policy.freeze()

@dataclass
class Completed:
    start_time:float; end_time:float; start_phase:int; end_phase:int; user_id:int; result:object

class Worker:
    def __init__(self,user_id,decoder,policy,request_tokens,records,lock,phase_info):
        self.user_id=user_id; self.decoder=decoder; self.policy=policy; self.request_tokens=request_tokens
        self.records=records; self.lock=lock; self.phase_info=phase_info; self.stop_event=threading.Event()
        self.thread=threading.Thread(target=self.run,daemon=True)
    def start(self): self.thread.start()
    def stop(self): self.stop_event.set()
    def join(self): self.thread.join()
    def run(self):
        req=0
        while not self.stop_event.is_set():
            with self.lock: phase=self.phase_info[0]
            prompt=TEST_PROMPTS[(self.user_id+req+phase)%len(TEST_PROMPTS)]
            start_time=time.perf_counter()
            result=self.decoder.generate(prompt,max_new_tokens=self.request_tokens,k=self.policy,do_sample=False,record_runtime_stats=False)
            end_time=time.perf_counter()
            with self.lock:
                end_phase=self.phase_info[0]
                self.records.append(Completed(start_time,end_time,phase,end_phase,self.user_id,result))
            req+=1

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--draft',default='meta-llama/Llama-3.2-1B'); p.add_argument('--target',default='meta-llama/Llama-3.2-3B'); p.add_argument('--device',default='cuda')
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=VARIANTS)
    p.add_argument('--schedule',nargs='+',type=int,default=[1,2,5,3,7,2,6,8,4,1,5,2,1])
    p.add_argument('--cycles',type=int,default=2); p.add_argument('--phase-seconds',type=float,default=5.0); p.add_argument('--request-tokens',type=int,default=24); p.add_argument('--train-tokens',type=int,default=80); p.add_argument('--reward-beta',type=float,default=0.0)
    p.add_argument('--out',default='dynamic_results.csv'); p.add_argument('--steps-out',default='dynamic_steps.csv'); a=p.parse_args()
    if torch.device(a.device).type=='mps' and max(a.schedule)>1:
        raise RuntimeError(
            "Concurrent decoding streams (>1) are not supported on MPS -- PyTorch's MPS backend "
            "crashes on concurrent multi-threaded GPU submission. Use device='cpu' for local "
            "concurrency testing on a Mac, or device='cuda' on the real target hardware."
        )
    dm,dt=load_model_and_tokenizer(a.draft,device=a.device); tm,tt=load_model_and_tokenizer(a.target,device=a.device)
    if dt.get_vocab()!=tt.get_vocab(): raise RuntimeError('Draft and target tokenizers differ.')
    decoder=SpeculativeDecoder(dm,tm,tt,device=a.device); decoder.generate(TRAIN_PROMPTS[0],max_new_tokens=8,k=3,do_sample=False)
    summary=[]; step_rows=[]
    for vi,variant in enumerate(a.variants):
        policy=FIXED[variant] if variant in FIXED else FACTORIES[variant](seed=vi,normalize_features=True,reward_fn=ThroughputRateReward(beta=a.reward_beta))
        if variant not in FIXED: train_policy(decoder,policy,a.train_tokens)
        records=[]; lock=threading.Lock(); phase_info=[0]; workers=[]; retired=[]; next_id=0
        phases=a.schedule*a.cycles
        for phase,target_users in enumerate(phases):
            with lock: phase_info[0]=phase
            # Increases are immediate. On decreases, excess workers finish their current
            # request before the measured phase begins. Retained workers keep running, so
            # the workload stays live while the measured concurrency matches users_target.
            while len(workers)<target_users:
                w=Worker(next_id,decoder,policy,a.request_tokens,records,lock,phase_info); next_id+=1; workers.append(w); w.start()
            if len(workers)>target_users:
                excess=workers[target_users:]; workers=workers[:target_users]
                for w in excess:w.stop()
                for w in excess:w.join()
                retired.extend(excess)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0=time.perf_counter(); time.sleep(a.phase_seconds)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t1=time.perf_counter()
            with lock:
                done=[r for r in records if t0 < r.end_time <= t1]
            steps=[s for r in done for s in r.result.steps]
            toks=sum(len(r.result.token_ids) for r in done)
            crossing=sum(r.start_time < t0 for r in done)
            summary.append(dict(variant=variant,phase=phase,users_target=target_users,completed_requests=len(done),crossing_requests=crossing,wall_time_s=t1-t0,tokens_per_second=toks/(t1-t0),avg_k=statistics.mean([s.k_requested for s in steps]) if steps else 0.0,acceptance_pct=100*sum(s.num_accepted for s in steps)/sum(s.k_requested for s in steps) if steps else 0.0))
            print(f'{variant:<10} phase={phase:02d} users={target_users:<2} completed={len(done):<3} TPS={summary[-1]["tokens_per_second"]:.2f} avg_k={summary[-1]["avg_k"]:.2f}',flush=True)
        all_workers=workers+retired
        for w in all_workers:w.stop()
        for w in all_workers:w.join()
        with lock: all_records=list(records)
        for r in all_records:
            for s in r.result.steps:
                step_rows.append(dict(variant=variant,start_phase=r.start_phase,end_phase=r.end_phase,user_id=r.user_id,step=s.step_index,k=s.k_requested,accepted=s.num_accepted,rejected=s.num_rejected,output_tokens=len(s.new_token_ids),draft_ms=1000*s.draft_time_s,verify_ms=1000*s.verify_time_s,compute_ms=1000*(s.draft_time_s+s.verify_time_s),round_wall_ms=1000*s.step_time_s,policy_us=1e6*s.policy_time_s))
    with open(a.out,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    with open(a.steps_out,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(step_rows[0])); w.writeheader(); w.writerows(step_rows)
    print(f'\nwrote {a.out}\nwrote {a.steps_out}')
if __name__=='__main__': main()
