from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
def intent_metrics(y_true,y_pred):
 labels=sorted(set(y_true)|set(y_pred)); p,r,f,s=precision_recall_fscore_support(y_true,y_pred,labels=labels,zero_division=0)
 return {'accuracy':float(accuracy_score(y_true,y_pred)),'macro_precision':float(p.mean()),'macro_recall':float(r.mean()),'macro_f1':float(f.mean()),'per_class':{label:{'precision':float(a),'recall':float(b),'f1':float(c),'support':int(d)} for label,a,b,c,d in zip(labels,p,r,f,s)},'labels':labels,'confusion_matrix':confusion_matrix(y_true,y_pred,labels=labels).tolist()}
def escalation_metrics(y_true,y_pred):
 labels=['AUTO_HANDLE','ESCALATE']; p,r,f,_=precision_recall_fscore_support(y_true,y_pred,labels=labels,average='binary',pos_label='ESCALATE',zero_division=0)
 return {'accuracy':float(accuracy_score(y_true,y_pred)),'precision':float(p),'recall':float(r),'f1':float(f),'false_negatives':int(sum(a=='ESCALATE' and b=='AUTO_HANDLE' for a,b in zip(y_true,y_pred))),'labels':labels,'confusion_matrix':confusion_matrix(y_true,y_pred,labels=labels).tolist()}
