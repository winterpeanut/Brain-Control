function [perms_train_dist,perms_test_dist]=LX_perms(n_perms,do_speech)

if nargin<2
    do_speech=1; %do not remove speech stimuli
end

[idx_train,idx_test]=get_folds;

if do_speech<1
    tmp_n_test=size(idx_test,1);
    tmp_n_train=size(idx_train,1);
    
    n_speech_test=tmp_n_test/6;
    n_speech_train=tmp_n_train/6;
    n_speech_tot=n_speech_test+n_speech_train;
    
    %new indices to distance matrices that have
    %already been devoided of speech stimuli, i.e.,
    idx_test_new=[];
    idx_train_new=[];
    if do_speech==0
        for i=1:size(idx_test,2)
            tmptest=idx_test(:,i);
            tmptest(tmptest<=n_speech_tot)=[];
            idx_test_new=cat(2,idx_test_new,tmptest-n_speech_tot);
            tmptrain=idx_train(:,i);
            tmptrain(tmptrain<=n_speech_tot)=[];
            idx_train_new=cat(2,idx_train_new,tmptrain-n_speech_tot);
        end
    else
        n_test=n_speech_test;
        n_train=n_speech_train;
        n_tot=n_test+n_train;
        
        thiscat=abs(do_speech);
        idx_first=(thiscat-1)*n_tot+1;
        idx_last=thiscat*n_tot;
        
        for i=1:size(idx_test,2)
            tmptest=idx_test(:,i);
            tmptest(tmptest<idx_first | tmptest>idx_last)=[];
            idx_test_new=cat(2,idx_test_new,tmptest-idx_first+1);
            tmptrain=idx_train(:,i);
            tmptrain(tmptrain<idx_first | tmptrain>idx_last)=[];
            idx_train_new=cat(2,idx_train_new,tmptrain-idx_first+1);
        end
    end
    idx_train=idx_train_new;
    idx_test=idx_test_new;
    disp(num2str(prctile(idx_train(:),[0 100])))
    disp(num2str(prctile(idx_test(:),[0 100])))
end

n_train_sounds=size(idx_train,1);
n_test_sounds=size(idx_test,1);
n_cvs=size(idx_train,2);

n_train_pairs=n_train_sounds;
n_test_pairs=n_test_sounds;

perms_train_dist=zeros([n_train_pairs n_cvs n_perms]);
perms_test_dist=zeros([n_test_pairs n_cvs n_perms]);

for perm=1:n_perms
    
    
    %%% prepare the test and training set object permutations
    perm_idx_test=zeros([n_test_sounds,n_cvs]);
    perm_idx_train=zeros([n_train_sounds,n_cvs]);
    for cv =1:n_cvs %permute the test sounds in each fold
        rnd_idx = randperm(n_test_sounds);
        perm_idx_test(:,cv)=idx_test(rnd_idx,cv);
    end
    for k=1:length(idx_test(:)) 
        %apply the test-set permutations to the corresponding sounds in the training set
        perm_idx_train(idx_train(:)==idx_test(k))=perm_idx_test(k);
    end
    
    
    %%% convert permutations of stimulus numbers
    %%% to permutations of column indices
    pos_idx_train=zeros([n_train_sounds,n_cvs]);
    pos_idx_test=zeros([n_test_sounds,n_cvs]);
    for cv=1:n_cvs
        for k=1:n_train_sounds
            pos_idx_train(k,cv)=find(idx_train(:,cv)==perm_idx_train(k,cv));
        end
        for k=1:n_test_sounds
            pos_idx_test(k,cv)=find(idx_test(:,cv)==perm_idx_test(k,cv));
        end
    end
    perms_train_dist(:,:,perm) = pos_idx_train;
    perms_test_dist(:,:,perm) = pos_idx_test;

  
end

end
% save PERMIdx.mat perms_test_dist perms_train_dist

function [idx_train,idx_test]=get_folds

idx_train=[1     4     1     1
    2     5     2     2
    3     6     3     3
    5     7     4     4
    6     8     5     6
    7     9     7     9
    8    10     8    10
    9    13    11    11
    10    14    12    12
    11    15    13    16
    12    16    14    17
    13    17    15    18
    14    18    17    19
    15    19    19    20
    16    20    20    21
    18    21    21    22
    22    22    23    23
    23    24    24    25
    24    26    25    26
    25    29    26    27
    27    30    27    28
    28    31    28    29
    30    32    29    32
    31    33    30    33
    32    34    31    34
    33    35    34    35
    36    37    35    36
    37    39    36    38
    38    40    37    39
    39    41    38    40
    40    42    41    41
    43    43    42    42
    44    44    43    44
    45    45    45    46
    46    47    46    47
    47    48    48    48
    50    49    49    49
    51    50    50    51
    53    52    51    52
    54    53    52    55
    55    54    53    58
    56    56    54    59
    57    57    55    60
    58    59    56    61
    59    61    57    65
    60    62    58    66
    61    63    60    68
    62    64    62    69
    63    65    63    70
    64    66    64    71
    65    67    67    72
    66    68    69    73
    67    69    71    74
    68    70    73    75
    70    72    75    76
    71    74    76    77
    72    75    77    78
    73    78    78    79
    74    79    80    80
    76    81    81    81
    77    82    82    82
    79    83    84    83
    80    84    85    84
    83    86    87    85
    85    89    88    86
    86    90    89    87
    87    91    90    88
    88    92    91    89
    93    93    92    90
    94    94    93    91
    95    95    95    92
    96    96    96    94
    97    97    97    98
    98    98    99   100
    99    99   101   101
    100   100   102   103
    102   101   103   104
    103   102   104   105
    104   106   105   108
    105   107   106   109
    106   108   107   111
    107   109   109   113
    108   110   110   114
    110   111   111   115
    112   112   112   116
    113   113   115   117
    114   114   116   118
    115   116   118   119
    117   117   120   120
    118   119   121   121
    119   121   123   122
    120   122   124   123
    122   123   125   125
    124   124   126   126
    126   125   127   127
    128   127   128   128
    129   130   129   129
    131   133   130   130
    132   134   131   131
    133   135   132   132
    134   137   134   133
    135   138   136   135
    136   139   137   136
    137   140   139   138
    138   141   140   139
    140   142   141   141
    143   143   142   142
    144   144   143   144
    145   145   147   145
    146   146   151   146
    148   147   152   147
    149   148   153   148
    150   149   154   149
    151   150   155   150
    153   151   157   152
    155   152   159   154
    156   153   160   156
    157   154   162   158
    158   155   163   161
    159   156   164   163
    160   157   165   164
    161   158   166   165
    162   159   167   166
    163   160   168   168
    164   161   169   169
    165   162   170   171
    166   167   171   172
    167   168   173   173
    169   170   174   174
    170   171   175   175
    172   172   176   176
    174   173   177   177
    175   177   178   179
    176   178   179   180
    178   180   180   181
    179   182   181   182
    181   183   182   183
    183   184   186   184
    184   185   187   185
    185   186   188   187
    186   187   189   189
    188   188   190   190
    191   189   191   191
    192   190   192   192
    193   194   193   193
    194   196   195   194
    195   197   197   195
    196   198   198   196
    197   199   199   200
    198   202   200   201
    199   203   201   202
    200   204   203   203
    201   206   204   205
    202   207   205   206
    204   208   206   207
    205   209   207   208
    209   210   208   210
    210   211   209   211
    211   212   212   212
    214   213   213   213
    215   215   214   214
    216   216   216   215
    217   217   217   218
    218   218   219   219
    220   219   220   221
    221   220   221   223
    222   222   222   224
    224   223   223   225
    225   224   225   226
    226   226   228   227
    227   227   229   229
    228   228   230   230
    229   232   231   231
    230   233   232   232
    231   234   233   233
    234   236   234   235
    235   237   235   237
    236   238   236   238
    237   239   238   239
    240   240   239   240
    241   243   241   241
    242   244   242   242
    243   245   243   244
    245   247   244   246
    246   248   245   247
    248   249   246   249
    250   250   247   250
    252   251   248   251
    253   253   249   252
    255   254   251   254
    256   255   252   256
    258   257   253   257
    259   259   254   258
    261   260   255   260
    262   261   256   261
    263   262   257   262
    264   263   258   264
    265   265   259   265
    266   269   260   266
    267   270   263   267
    268   271   264   268
    269   272   266   271
    270   273   267   272
    273   274   268   273
    274   276   269   275
    275   277   270   276
    276   278   271   278
    277   279   272   279
    279   280   274   280
    280   281   275   282
    281   283   277   283
    282   284   278   284
    284   285   281   285
    285   286   282   286
    287   287   283   287
    288   288   286   288];

idx_test=[     4     1     6     5
    17     2     9     7
    19     3    10     8
    20    11    16    13
    21    12    18    14
    26    23    22    15
    29    25    32    24
    34    27    33    30
    35    28    39    31
    41    36    40    37
    42    38    44    43
    48    46    47    45
    49    51    59    50
    52    55    61    53
    69    58    65    54
    75    60    66    56
    78    71    68    57
    81    73    70    62
    82    76    72    63
    84    77    74    64
    89    80    79    67
    90    85    83    93
    91    87    86    95
    92    88    94    96
    101   103    98    97
    109   104   100    99
    111   105   108   102
    116   115   113   106
    121   118   114   107
    123   120   117   110
    125   126   119   112
    127   128   122   124
    130   129   133   134
    139   131   135   137
    141   132   138   140
    142   136   144   143
    147   163   145   151
    152   164   146   153
    154   165   148   155
    168   166   149   157
    171   169   150   159
    173   174   156   160
    177   175   158   162
    180   176   161   167
    182   179   172   170
    187   181   183   178
    189   191   184   186
    190   192   185   188
    203   193   194   197
    206   195   196   198
    207   200   202   199
    208   201   210   204
    212   205   211   209
    213   214   215   216
    219   221   218   217
    223   225   224   220
    232   229   226   222
    233   230   227   228
    238   231   237   234
    239   235   240   236
    244   241   250   243
    247   242   261   245
    249   246   262   248
    251   252   265   253
    254   256   273   255
    257   258   276   259
    260   264   279   263
    271   266   280   269
    272   267   284   270
    278   268   285   274
    283   275   287   277
    286   282   288   281];
end

