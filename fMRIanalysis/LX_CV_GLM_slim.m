function out=LX_CV_GLM_slim(Preds,Y_train,Y_test,lambdas)

Y_train = zscore(Y_train);
Y_test = zscore(Y_test);



for ipred=1:size(Preds,1)
    
    
    Xtrain=Preds{ipred,1};
    
    [BetaTrain,~,rsqtmp]=LX_BLG_GLM_inv(Xtrain,Y_train,0,1,lambdas); %Pred is already the prediction
    beta_train{ipred}=BetaTrain;
    
    
    fdemean=@(x) bsxfun(@minus,x,mean(x));
    fregrdem=@(x) cat(2,ones(size(x(:,1,:))),fdemean(x)); %demean and add intercept
    
    Xtest=fregrdem(Preds{ipred,2});
    
    Pred=mtimesx(Xtest,BetaTrain);
    
    
    
    
   
    SSTtest=sum(bsxfun(@minus,Y_test,mean(Y_test)).^2);
    SSEtest=sum(bsxfun(@minus,Y_test,Pred).^2,1);
    tmp=squeeze(1-SSEtest./SSTtest);
        %                 tmp=BLGmx_corr2(tmp_Y_test_perms(:,:,:,:,end),Pred);
   
    
    %%
    RSQcv{ipred}=tmp;
    R2_ols{ipred}=rsqtmp;
    n_preds(ipred)=size(BetaTrain,1)-1;
    

    %%
    
  
end




%%% Prepare output


out.RSQcv=RSQcv;
out.R2_ols=R2_ols;
out.results.n_preds=n_preds;
out.results.beta_train=beta_train;


end

