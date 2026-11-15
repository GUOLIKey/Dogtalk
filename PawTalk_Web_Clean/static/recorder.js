/* Microphone lifecycle; injectable browser APIs allow permission/race tests. */
(function(root){
class HoldRecorder{
 constructor({getUserMedia,Recorder,onState=()=>{},onBlob=()=>{},onError=()=>{},maxMs=59000,maxBytes=64*1024*1024}){Object.assign(this,{getUserMedia,Recorder,onState,onBlob,onError,maxMs,maxBytes});this.state='idle';this.held=false;this.discard=false;this.stream=null;this.recorder=null;this.timer=null;this.chunks=[];}
 get active(){return this.state!=='idle';}
 emit(state){this.state=state;this.onState(state);}
 release(){clearTimeout(this.timer);this.timer=null;if(this.stream)for(const track of this.stream.getTracks())track.stop();this.stream=null;this.recorder=null;this.held=false;this.emit('idle');}
 async begin(){if(this.active)return;this.held=true;this.discard=false;this.chunks=[];this.emit('requesting');try{this.stream=await this.getUserMedia({audio:{echoCancellation:false,noiseSuppression:false,autoGainControl:false},video:false});if(!this.held){this.release();return;}const mime=['audio/webm;codecs=opus','audio/mp4','audio/ogg;codecs=opus'].find(m=>this.Recorder.isTypeSupported?.(m));const recorder=new this.Recorder(this.stream,mime?{mimeType:mime}:undefined);this.recorder=recorder;let size=0;
 recorder.ondataavailable=e=>{if(e.data?.size){size+=e.data.size;this.chunks.push(e.data);if(size>this.maxBytes){this.end(true);this.onError(Error('录音超过大小限制，请缩短后重试。'));}}};
 recorder.onerror=()=>{this.discard=true;try{if(recorder.state!=='inactive')recorder.stop();}catch{}this.release();this.onError(Error('录音失败，请检查麦克风或改用上传文件。'));};
 recorder.onstop=()=>{const cancelled=this.discard;const blob=new Blob(this.chunks,{type:recorder.mimeType||this.chunks[0]?.type||'audio/webm'});this.chunks=[];this.release();if(!cancelled&&blob.size)this.onBlob(blob);};
 recorder.start(250);this.emit('recording');this.timer=setTimeout(()=>this.end(),this.maxMs);
 }catch(error){this.discard=true;this.release();this.onError(error);}}
 end(discard=false){this.held=false;this.discard=this.discard||discard;if(this.recorder&&this.recorder.state!=='inactive'){this.emit('stopping');this.recorder.stop();}}
}
if(typeof module!=='undefined'&&module.exports)module.exports=HoldRecorder;else root.HoldRecorder=HoldRecorder;
})(globalThis);
