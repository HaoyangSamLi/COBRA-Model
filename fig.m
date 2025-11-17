%% fig 1
close all
clear
clc
load("Params\Params.mat",'SLIB','RowName','GeoParam');
% Amount sorted by cities and types, 2035
ColName = {'City','Total','LFP','LMO','NCM-L','NCM-M','NCM-H','NCA','Capacity2023'};
capacity = GeoParam.Ref(:,3);
SLIB = SLIB(:,1:16,1:6);
for t = 1:5:16
    camt = squeeze(SLIB(:,t,:));
    writecell([ColName;RowName(:,1),num2cell([sum(camt,2),camt,capacity])],sprintf("C:\\Users\\昊阳\\Desktop\\Alpha\\02 Figure\\FigureProject\\fig1\\CityAmount%d.csv",t+2019));
end
% Amount sorted by time and types
TLIB = squeeze(sum(SLIB,1));
RowName = linspace(2020,2035,16)';
ColName = {'Time','LFP','LMO','NCM-L','NCM-M','NCM-H','NCA'};
writecell([ColName;num2cell(RowName),num2cell(TLIB)],"C:\Users\昊阳\Desktop\Alpha\02 Figure\FigureProject\fig1\AnnualAmount.csv");
% Amount sorted by types and regions, major types
ColName = {'Type','North','Northeast','East','Central','South','Southwest','Northwest'};
ramt = [];
for k = 1:6
    ramt = [ramt;sum(SLIB(1:36,:,k),'all'),sum(SLIB(37:72,:,k),'all'),sum(SLIB(73:149,:,k),'all'),sum(SLIB(150:193,:,k),'all'),sum(SLIB(194:232,:,k),'all'),sum(SLIB(233:286,:,k),'all'),sum(SLIB(287:337,:,k),'all')];
end
RowName = {'LFP','NCM-H','NCM-M','Others'}';
ramt = [ramt([1,5,4],:);sum(ramt([2,3,6],:),1)];
writecell([ColName;RowName,num2cell(ramt)],"C:\Users\昊阳\Desktop\Alpha\02 Figure\FigureProject\fig1\RegionalAmount.csv");
RowName = {'LFP','LMO','NCM-L','NCM-M','NCM-H','NCA'}';
for t = 1:5:16
    ramt = [];
    for k = 1:6
        ramt = [ramt;sum(SLIB(1:36,t,k),'all'),sum(SLIB(37:72,t,k),'all'),sum(SLIB(73:149,t,k),'all'),sum(SLIB(150:193,t,k),'all'),sum(SLIB(194:232,t,k),'all'),sum(SLIB(233:286,t,k),'all'),sum(SLIB(287:337,t,k),'all')];
    end
    writecell([ColName;RowName,num2cell(ramt)],sprintf("C:\\Users\\昊阳\\Desktop\\Alpha\\02 Figure\\FigureProject\\fig1\\RegionalAmount%d.csv",t+2019));
end
%% fig 2
close all
clear
clc
% Cost & emission level on city-level, every 5 years
load("Params\RecParam@Scen1@Vars0@Pert0%.mat");
load("Params\Params.mat",'RowName');
RC = PT + LD + LB + MT + IS + RD + GA + RM + UT + WD;
RE = PC + UM + WT;
for t = 1:5:16
    rc = squeeze(RC(:,t,:));
    re = squeeze(RE(:,t,:));
    ColName = {'City','Cost LFP','Cost LMO','Cost NCM-L','Cost NCM-M','Cost NCM-H','Cost NCA','Emission LFP','Emission LMO','Emission NCM-L','Emission NCM-M','Emission NCM-H','Emission NCA'};
    writecell([ColName;RowName(:,1),num2cell([rc,re])],sprintf("C:\\Users\\昊阳\\Desktop\\Alpha\\02 Figure\\FigureProject\\fig2\\CostEmissionLevel%d.csv",t+2019));
end
% Cost & emission trend by cities and regions, every 5 years
rowname = [repmat({'North'},36,1);repmat({'Northeast'},36,1);repmat({'East'},77,1);repmat({'Central'},44,1);repmat({'South'},39,1);repmat({'Southwest'},54,1);repmat({'Northwest'},51,1)];
RowName = [rowname,RowName(:,2)];
for k = 1:6
    rc = RC(:,1:5:16,k);
    re = RE(:,1:5:16,k);
    ColName = {'Region','City','Cost 2020','Cost 2025','Cost 2030','Cost 2035','Emission 2020','Emission 2025','Emission 2030','Emission 2035'};
    writecell([ColName;RowName,num2cell([rc,re])],sprintf("C:\\Users\\昊阳\\Desktop\\Alpha\\02 Figure\\FigureProject\\fig2\\CostEmissionTrend%d.csv",k));
end
% Cost & emission breakdown, every 5 years
for t = 1:5:16
    pt = mean(PT(:,t,:),1);
    ld = mean(LD(:,t,:),1);
    lb = mean(LB(:,t,:),1);
    mt = mean(MT(:,t,:),1);
    is = mean(IS(:,t,:),1);
    rd = mean(RD(:,t,:),1);
    ga = mean(GA(:,t,:),1);
    bt = mean(BT(:,t,:),1);
    rm = mean(RM(:,t,:),1);
    ut = mean(UT(:,t,:),1);
    wd = mean(WD(:,t,:),1);
    pc = mean(PC(:,t,:),1);
    um = mean(UM(:,t,:),1);
    wt = mean(WT(:,t,:),1);
    cost = pt + ld + lb + mt + is + rd + ga + rm + ut + wd;
    emission = pc + um + wt;
    data = squeeze([cost;pt;ld;lb;mt;is;rd;ga;rm;ut;wd;emission;pc;um;wt]);
    ColName = {'Breakdown','LFP','LMO','NCM-L','NCM-M','NCM-H','NCA'};
    RowName = {'Cost','Plant','Land','Labor','Maintenance','Insurance','R&D','G&A','Raw materials','Utilities','Waste disposal','Emission','Process','Upstream material','Waste treatment'}';
    writecell([ColName;RowName,num2cell(data)],sprintf("C:\\Users\\昊阳\\Desktop\\Alpha\\02 Figure\\FigureProject\\fig2\\CostEmissionBreakdown%d.csv",t+2019));
end
%% fig 3
close all
clear
clc
step = 5;
seq = 1:step+1;
% Route parsing and capacity by 2035, every 20% percent GHG savings
load("Params\Params.mat",'RowName','GeoParam');
loc = GeoParam.Ref(:,4:5);
load("Solutions\Sol@Scen1@Vars0@Pert0%@Mode1@Gaps5.mat");
capacity = [];
RowName = RowName(:,2);
for i = seq
    flow = sum(solution(i).flow,4);
    amount = sum(solution(i).amount,3);
    ff = sum(flow,3);
    route = {'Origin','Destination','OX','OY','DX','DY','Total','Amount2020','Amount2025','Amount2030','Amount2035'};
    for o = 1:337
        for d = 1:337
            if ff(o,d) > 0
                data = [loc(o,:),loc(d,:),ff(o,d),flow(o,d,1),flow(o,d,6),flow(o,d,11),flow(o,d,16)];
                route = [route;RowName(o),RowName(d),num2cell(data)];
            end
        end
    end
    writecell(route,sprintf("C:\\Users\\昊阳\\Desktop\\Alpha\\02 Figure\\FigureProject\\fig3\\FlowAmount%d.csv",i));
    capacity = [capacity,amount(:,1),amount(:,6),amount(:,11),amount(:,16)];
end
load("Params\Params.mat",'RowName')
RowName = RowName(:,1);
ColName = {'City','X','Y','0%@2020','0%@2025','0%@2030','0%@2035','20%@2020','20%@2025','20%@2030','20%@2035','40%@2020','40%@2025','40%@2030','40%@2035','60%@2020','60%@2025','60%@2030','60%@2035','80%@2020','80%@2025','80%@2030','80%@2035','100%@2020','100%@2025','100%@2030','100%@2035'};
writecell([ColName;RowName,num2cell([loc,capacity])],"C:\Users\昊阳\Desktop\Alpha\02 Figure\FigureProject\fig3\CityCapacity.csv");
% Provincial flow
data = [];
Province = importdata("Data\Province.xlsx");
Province = Province(:,2);
noc = [1,1,11,11,12,14,9,13,1,13,11,16,9,11,16,17,13,14,21,14,4,1,21,9,16,7,10,14,8,5,14];
soc = cumsum(noc);
A = zeros(31,337);
A(1,1) = 1;
for i = 2:31
    A(i,soc(i-1)+1:soc(i)) = ones(1,noc(i));
end
k = 12; % Parameter quick search, user defined
listV = [];
for i = seq
    flow = sum(solution(i).flow,4);
    amount = sum(solution(i).amount,3);
    flow = sum(flow,3);
    flow = A*flow*A';
    inflow = [linspace(1,31,31)',sum(flow,1)'];
    outflow = [linspace(1,31,31)',sum(flow,2)];
    innout = (inflow + outflow)/2;
    innout = sortrows(innout,2,'descend');
    listV = [listV;innout(1:k,1)];
end
listV = unique(listV);
for i = seq
    flow = sum(solution(i).flow,4);
    amount = sum(solution(i).amount,3);
    flow = sum(flow,3);
    flow = A*flow*A';
    m = length(listV); % A genius switch algorithim
    flow = [flow,ones(31,1)*2023310481;ones(1,31)*2023310481,2023310481];
    for j = 1:m
        flow(listV(j),32) = j;
        flow(32,listV(j)) = j;
    end
    flow = sortrows(flow,32);
    flow = flow';
    flow = sortrows(flow,32);
    flow = flow';
    flow = flow(1:31,1:31);
    namelist = [];
    for j = 1:m
        namelist = [namelist;Province(listV(j))];
    end
    namelist = [namelist;'Others'];
    B = [eye(m),zeros(m,31-m);zeros(1,m),ones(1,31-m)];
    flow = B*flow*B';
    writecell(['Province',namelist';namelist,num2cell(flow/1e6)],sprintf("C:\\Users\\昊阳\\Desktop\\Alpha\\02 Figure\\FigureProject\\fig3\\ProvincialFlow%d.csv",i));
end
% Separated frieght volume, sorted by time and by boundaries
A = eye(337);
noc = [1,1,11,11,12,14,9,13,1,13,11,16,9,11,16,17,13,14,21,14,4,1,21,9,16,7,10,14,8,5,14];
B = [];
for i = 1:31
    B = blkdiag(B,ones(noc(i)));
end
B = B - A;
C = ones(337,337) - B - A;
for i = seq
    flow = sum(solution(i).flow,4);
    freightvolume = [];
    for t = 1:16
        ff = flow(:,:,t);
        freightvolume = [freightvolume;sum(ff.*A,"all"),sum(ff.*B,"all"),sum(ff.*C,"all")];
    end
    ColName = {'Time','Local','Intra','Inter'};
    RowName = num2cell([2020:2035]');
    writecell([ColName;RowName,num2cell(freightvolume)],sprintf("C:\\Users\\昊阳\\Desktop\\Alpha\\02 Figure\\FigureProject\\fig3\\SeparatedTransportVolume%d.csv",i));
end
for t = 1:5:16
    freightvolume = [];
    for i = seq
        flow = sum(solution(i).flow,4);
        flow = flow(:,:,t);
        freightvolume = [freightvolume;sum(flow.*A,"all"),sum(flow.*B,"all"),sum(flow.*C,"all")];
    end
    ColName = {'Index','Local','Intra','Inter'};
    RowName = num2cell(seq');
    writecell([ColName;RowName,num2cell(freightvolume)],sprintf("C:\\Users\\昊阳\\Desktop\\Alpha\\02 Figure\\FigureProject\\fig3\\SeparatedTransportVolume%d.csv",t+2019));
end
% Separated frieght work, sorted by time and by boundaries
A = eye(337);
noc = [1,1,11,11,12,14,9,13,1,13,11,16,9,11,16,17,13,14,21,14,4,1,21,9,16,7,10,14,8,5,14];
B = [];
for i = 1:31
    B = blkdiag(B,ones(noc(i)));
end
B = B - A;
C = ones(337,337) - B - A;
load("Params\Params.mat","LogParam");
D = LogParam.Dist;
for i = seq
    flow = sum(solution(i).flow,4);
    freightvolume = [];
    for t = 1:16
        ff = flow(:,:,t).*D;
        freightvolume = [freightvolume;sum(ff.*A,"all"),sum(ff.*B,"all"),sum(ff.*C,"all")];
    end
    ColName = {'Time','Local','Intra','Inter'};
    RowName = num2cell([2020:1:2035]');
    writecell([ColName;RowName,num2cell(freightvolume)],sprintf("C:\\Users\\昊阳\\Desktop\\Alpha\\02 Figure\\FigureProject\\fig3\\SeparatedTransportWork%d.csv",i));
end
for t = 1:5:16
    freightvolume = [];
    for i = seq
        flow = sum(solution(i).flow,4);
        ff = flow(:,:,t).*D;
        freightvolume = [freightvolume;sum(ff.*A,"all"),sum(ff.*B,"all"),sum(ff.*C,"all")];
    end
    ColName = {'Index','Local','Intra','Inter'};
    RowName = num2cell(seq');
    writecell([ColName;RowName,num2cell(freightvolume)],sprintf("C:\\Users\\昊阳\\Desktop\\Alpha\\02 Figure\\FigureProject\\fig3\\SeparatedTransportWork%d.csv",t+2019));
end
%% fig 4
close all
clear
clc
load("Solutions\Sol@Scen1@Vars0@Pert0%@Mode1@Gaps100.mat");
step = 100;
seq = 1:step+1;
% Overall performance
totalcost = [];
totalemission = [];
for i = 1:step+1
    totalcost = [totalcost;sum(solution(i).cost,'all')];
    totalemission = [totalemission;sum(solution(i).emission,'all')];
end
ColName = {'Index','Cost','Emission'};
RowName = num2cell(seq');
data = [totalcost(seq,:),totalemission(seq,:)];
writecell([ColName;RowName,num2cell(data)],"C:\Users\昊阳\Desktop\Alpha\02 Figure\FigureProject\fig4\NetworkPerformance.csv");
% Abatement cost curve
reduction  = ones(step + 1,1)*totalemission(1,1) - totalemission;
abatementcost =  totalcost - ones(step + 1,1)*totalcost(1,1);
ColName = {'Index','Reduction','Abatement cost'};
writecell([ColName;num2cell(linspace(1,step+1,step+1)'),num2cell([reduction,abatementcost./reduction])],"C:\Users\昊阳\Desktop\Alpha\02 Figure\FigureProject\fig4\AbatementCost.csv");
% Bridge
city = 337;
time = 16;
type = 6;
s = 1;
step = 5;
seq = 1:step+1;
load("Params\Params.mat")
load("Params\RecParam@Scen1@Vars0@Pert0%.mat")
load(sprintf("Solutions\\Sol@Scen1@Vars0@Pert0%%@Mode1@Gaps%d.mat",step));
fct = ScenParam(9,s);
fcp = ScenParam(10,s);
cpd = zeros(1,time);
if fcp ~= 0
    for i = 1:time
        cpd(i) = 7.10*exp(fcp*(i - 1));
    end
end
costdata = [];
emissiondata = [];
RC = zeros(city,city,time,type);
RE = zeros(city,city,time,type);
FLAG = zeros(city,city,time,type);
CPD = zeros(city,city,time,type);
noc = [1,1,11,11,12,14,9,13,1,13,11,16,9,11,16,17,13,14,21,14,4,1,21,9,16,7,10,14,8,5,14];
A = [];
for i = 1:31
    A = blkdiag(A,ones(noc(i)));
end
B = ones(337,337) - A; % Block matrix
for n = 1:city
    RE(n,:,:,:) = PC + UM + WT;
    CPD(n,:,:,:) = repmat(cpd,city,1,type);
    CTR = (PC + UM + WT).*repmat(cpd,city,1,type);
    flag = fct*(RV/1.13 - BT/1.13 - RM/1.13 - Elec/1.13 - Gas/1.09 - Water/1.09 - WD - squeeze(PS(n,:,:,:)) - LB - MT - IS - RD - GA - PT - LD - CTR) > 0;
    FLAG(n,:,:,:) = flag;
    RC(n,:,:,:) = PT + LD + RM + UT + WD + LB + MT + IS + RD + GA + max((RV - BT - RM - Elec)/1.13*0.13 - (Gas + Water)/1.09*0.09,0) + fct*(RV/1.13 - BT/1.13 - RM/1.13 - Elec/1.13 - Gas/1.09 - Water/1.09 - WD - LB - MT - IS - RD - GA - PT - LD).*flag + CTR;
    SC(n,:,:,:) = SC(n,:,:,:) - fct*(SC(n,:,:,:) + CPD(n,:,:,:).*SE(n,:,:,:)).*FLAG(n,:,:,:) + CPD(n,:,:,:).*SE(n,:,:,:);
    TC(n,:,:,:) = TC(n,:,:,:) - fct*(TC(n,:,:,:) + CPD(n,:,:,:).*TE(n,:,:,:)).*FLAG(n,:,:,:) + CPD(n,:,:,:).*TE(n,:,:,:);
end
flow = solution(1).flow;
rc0 = RC.*flow;
re0 = RE.*flow;
sc0 = SC.*flow;
se0 = SE.*flow;
tc0 = TC.*flow;
te0 = TE.*flow;
for i = seq
    flow = solution(i).flow;
    rc1 = RC.*flow;
    re1 = RE.*flow;
    sc1 = SC.*flow;
    se1 = SE.*flow;
    tc1 = TC.*flow;
    te1 = TE.*flow;
    drc = rc1 - rc0;
    dre = re1 - re0;
    dsc = sc1 - sc0;
    dse = se1 - se0;
    dtc = tc1 - tc0;
    dte = te1 - te0;
    data = [sum(rc0+sc0+tc0,"all");sum(dsc,"all");sum(dtc.*A,"all");sum(dtc.*B,"all");sum(drc,"all")];
    costdata = [costdata;data;-1*sum(data)];
    data = [sum(re0+se0+te0,"all");sum(dse,"all");sum(dte.*A,"all");sum(dte.*B,"all");sum(dre,"all")];
    emissiondata = [emissiondata;data;-1*sum(data)];
end
costdata = [costdata;0];
emissiondata = [emissiondata;0];
ColName = {'Cost','Emission'};
writecell([ColName;num2cell([costdata,emissiondata])],"C:\Users\昊阳\Desktop\Alpha\02 Figure\FigureProject\fig4\Bridge.csv");
%% fig 5
close all
clear
clc
load("Solutions\Sol@Scen1@Vars0@Pert0%@Mode1@Gaps100.mat");
step = 100;
seq = 1:step+1;
% Demand-goal-weighted capacity
load("Params\Params.mat",'RowName','GeoParam','SLIB');
ColName = {'城市','City','Participation','Variance','Correlation','Current state','Maximum','Rmax','Minimum','Rmin'};
SLIB = sum(SLIB(:,1:16,1:6),3);
weight = sum(SLIB,1)/sum(SLIB,"all");
capacity = [];
correlation = [];
goal = (seq-1)/100;
for i = seq
    amount = sum(solution(i).amount,3);
    %capacity = [capacity,amount*weight'*(i-1)/100]; % demand-goal-weighted
    capacity = [capacity,amount(:,end)];
    ColName = [ColName,cellstr(sprintf("R%d",i-1))];
end
for i = 1:337
    correlation = [correlation;corr(capacity(i,:)',goal')];
end
participation = mean(capacity,2);
variance = var(capacity,0,2);
[maximum,rmax] = max(fliplr(capacity),[],2);
[minimum,rmin] = min(fliplr(capacity),[],2);
rmax = (101 - rmax)/100;
rmin = (101 - rmin)/100;
city = participation ~= 0;
participation = normalize(participation(city),'range');
variance = normalize(variance(city),'range');
capacity = capacity(city,:);
correlation = correlation(city,:);
current = GeoParam.Ref(:,3);
current = current(city);
maximum = maximum(city,:);
minimum = minimum(city,:);
rmax = rmax(city,:);
rmin = rmin(city,:);
RowName = RowName(city,:);
writecell([ColName;RowName,num2cell([participation,variance,correlation,current,maximum,rmax,minimum,rmin,capacity])],"C:\Users\昊阳\Desktop\Alpha\02 Figure\FigureProject\fig5\Metrics.csv");
% City capacity by year and network
load("Solutions\Sol@Scen1@Vars0@Pert0%@Mode1@Gaps5.mat");
RowName = RowName(:,2);
step = 5;
seq = 1:step+1;
ColName = num2cell(seq);
for j = 1:size(capacity,1)
    data = [];
    for i = seq
        amount = sum(solution(i).amount,3);
        amount = amount(city,:)';
        data = [data,amount(:,j)];
    end
    writecell([RowName{j},ColName;num2cell([2020:2035]'),num2cell(data)],sprintf("C:\\Users\\昊阳\\Desktop\\Alpha\\02 Figure\\FigureProject\\fig5\\Capacity%s.csv",RowName{j}));
end
%% Sensitivity analysis and scenario comparison
close all
clear
clc
s = 1;
m = 1;
g = 5;
for v = [28,30,31,32,34,36,38,39,41,44] % Specific
    for p = [-0.10,0.10]
        pair = parse(s,v,p,m,g);
    end
end
% Scenario comparison
v = 0;
p = 0;
m = 1;
for s = 15
    for g = [5,100]
        parse(s,v,p,m,g);
    end
end
% Sensitivity analysis
elasticity = [];
pair = [];
s = 1;
m = 1;
g = 5;
for v = 1:45 % Specific for COBRA V5.2
    pair = [];
    for p = [-0.10,0.10]
        pair = [pair;parse(s,v,p,m,g)];
    end
    if v >= 30 && v <= 34
        pair(2,:) = zeros(1,12);
    end
    pair = [pair;max(abs(pair),[],1)];
    elasticity = [elasticity;reshape(pair,1,[])];
end
ColName = {'Cost1-10%','Cost1+10%','DeltaC1', ...
    'Cost2-10%','Cost2+10%','DeltaC2', ...
    'Cost3-10%','Cost3+10%','DeltaC3', ...
    'Cost4-10%','Cost4+10%','DeltaC4', ...
    'Cost5-10%','Cost5+10%','DeltaC5', ...
    'Cost6-10%','Cost6+10%','DeltaC6', ...
    'Emission1-10%','Emission1+10%','DeltaE1', ...
    'Emission2-10%','Emission2+10%','DeltaE2', ...
    'Emission3-10%','Emission3+10%','DeltaE3', ...
    'Emission4-10%','Emission4+10%','DeltaE4', ...
    'Emission5-10%','Emission5+10%','DeltaE5', ...
    'Emission6-10%','Emission6+10%','DeltaE6'};
RowName = {'Labor wage', ...
    'Land price', ...
    'Energy charge', ...
    'Capacity charge', ...
    'Gas price', ...
    'Water price', ...
    'Precursor price', ...
    'Lithium price', ...
    'Nickel price', ...
    'Cobalt price', ...
    'Manganese price', ...
    'Chemicals price', ...
    'Byproducts price', ...
    'Disposal charge', ...
    'Equipment price', ...
    'Vehicle price', ...
    'Driver wage', ...
    'Diesel price', ...
    'Toll charge', ...
    'Transport insurance', ...
    'Packaging price', ...
    'Storage capital', ...
    'Storage insurance', ...
    'Labor input', ...
    'Land use', ...
    'Electricity consumption', ...
    'Power capacity', ...
    'Gas consumption', ...
    'Water consumption', ...
    'Precursor yield', ...
    'Lithium yield', ...
    'Cobalt yield', ...
    'Manganese yield', ...
    'Byproducts yield', ...
    'Waste yield', ...
    'Transport distance', ...
    'Annual mileage', ...
    'Vehicle payload', ...
    'Fuel consumption', ...
    'Storage labor', ...
    'Storage electricity', ...
    'Storage power', ...
    'Storage water', ...
    'Storage capacity', ...
    'Storage duration'}'; 
for i = 1:12
    data = [RowName,num2cell(elasticity(:,3*i-2:3*i))];
    data = sortrows(data,4,'ascend');
    if i <= 6
        writecell(['Elasticity',ColName(:,3*i-2:3*i);data],sprintf("C:\\Users\\昊阳\\Desktop\\Alpha\\02 Figure\\FigureProject\\figs\\ElasticityC%d.csv",i));
    else
        writecell(['Elasticity',ColName(:,3*i-2:3*i);data],sprintf("C:\\Users\\昊阳\\Desktop\\Alpha\\02 Figure\\FigureProject\\figs\\ElasticityE%d.csv",i-6));
    end
end
% function defination
function elasticity = parse(s,v,p,m,g)
path = sprintf("C:\\Users\\昊阳\\Desktop\\Alpha\\02 Figure\\FigureProject\\figs\\Figs@Scen%d@Vars%d@Pert%d%%@Mode%d@Gaps%d",s,v,p*100,m,g);
if exist(path,'file') == 0
    mkdir(path);
end
if g == 5
    step = 5; % not global settings, in accordance with 2020-2035
    seq = 1:step+1;
    load("Params\Params.mat",'SLIB','RowName','GeoParam');
    SLIB = SLIB(:,1:16,1:6);
    RowName = RowName(:,2);
    loc = GeoParam.Ref(:,4:5);
    load("Params\RecParam@Scen1@Vars0@Pert0%.mat",'BT'); % baseline
    bat0 = sum(SLIB.*BT,"all");
    load("Solutions\Sol@Scen1@Vars0@Pert0%@Mode1@Gaps5.mat"); % baseline
    baseline = solution;
    load(sprintf("Params\\RecParam@Scen%d@Vars%d@Pert%d%%.mat",s,v,p*100),'BT');
    bat = sum(SLIB.*BT,"all");
    load(sprintf("Solutions\\Sol@Scen%d@Vars%d@Pert%d%%@Mode%d@Gaps%d.mat",s,v,p*100,m,g));
    capacity = [];
    elasticity = [];
    for i = seq
        flow = sum(solution(i).flow,4);
        amount = sum(solution(i).amount,3);
        delta_cost = (sum(solution(i).cost,"all")-sum(baseline(i).cost,"all")+bat-bat0)/(sum(baseline(i).cost,"all")+bat0); % Using total cost, incorporating battery purchasing cost
        delta_emission = (sum(solution(i).emission,"all")-sum(baseline(i).emission,"all"))/sum(baseline(i).emission,"all");
        ff = sum(flow,3);
        route = {'Origin','Destination','OX','OY','DX','DY','Total','Amount2020','Amount2025','Amount2030','Amount2035'};
        for o = 1:337
            for d = 1:337
                if ff(o,d) > 0
                    data = [loc(o,:),loc(d,:),ff(o,d),flow(o,d,1),flow(o,d,6),flow(o,d,11),flow(o,d,16)];
                    route = [route;RowName(o),RowName(d),num2cell(data)];
                end
            end
        end
        writecell(route,sprintf("C:\\Users\\昊阳\\Desktop\\Alpha\\02 Figure\\FigureProject\\figs\\Figs@Scen%d@Vars%d@Pert%d%%@Mode%d@Gaps%d\\FlowAmount%d@Scen%d@Vars%d@Pert%d%%@Mode%d@Gaps%d.csv",s,v,p*100,m,g,i,s,v,p*100,m,g));
        capacity = [capacity,amount(:,1),amount(:,6),amount(:,11),amount(:,16)];
        elasticity = [elasticity;delta_cost,delta_emission];
    end
    elasticity = reshape(elasticity,1,[]);
    load("Params\Params.mat",'RowName')
    RowName = RowName(:,1);
    ColName = {'City','X','Y','0%@2020','0%@2025','0%@2030','0%@2035','20%@2020','20%@2025','20%@2030','20%@2035','40%@2020','40%@2025','40%@2030','40%@2035','60%@2020','60%@2025','60%@2030','60%@2035','80%@2020','80%@2025','80%@2030','80%@2035','100%@2020','100%@2025','100%@2030','100%@2035'};
    writecell([ColName;RowName,num2cell([loc,capacity])],sprintf("C:\\Users\\昊阳\\Desktop\\Alpha\\02 Figure\\FigureProject\\figs\\Figs@Scen%d@Vars%d@Pert%d%%@Mode%d@Gaps%d\\CityCapacity@Scen%d@Vars%d@Pert%d%%@Mode%d@Gaps%d.csv",s,v,p*100,m,g,s,v,p*100,m,g));
    writecell([ColName;RowName,num2cell([loc,capacity])],sprintf("C:\\Users\\昊阳\\Desktop\\Alpha\\02 Figure\\FigureProject\\figs\\Capacity.csv"));
elseif g == 100
    step = 100;
    seq = 1:step+1;
    load(sprintf("Solutions\\Sol@Scen%d@Vars%d@Pert%d%%@Mode%d@Gaps%d.mat",s,v,p*100,m,g));
    % Overall performance
    totalcost = [];
    totalemission = [];
    for i = 1:step+1
        totalcost = [totalcost;sum(solution(i).cost,'all')];
        totalemission = [totalemission;sum(solution(i).emission,'all')];
    end
    ColName = {'Index','Cost','Emission'};
    RowName = num2cell(seq');
    data = [totalcost(seq,:),totalemission(seq,:)];
    writecell([ColName;RowName,num2cell(data)],sprintf("C:\\Users\\昊阳\\Desktop\\Alpha\\02 Figure\\FigureProject\\figs\\Figs@Scen%d@Vars%d@Pert%d%%@Mode%d@Gaps%d\\NetworkPerformance@Scen%d@Vars%d@Pert%d%%@Mode%d@Gaps%d.csv",s,v,p*100,m,g,s,v,p*100,m,g));
    % Abatement cost curve
    reduction  = ones(step + 1,1)*totalemission(1,1) - totalemission;
    abatementcost =  totalcost - ones(step + 1,1)*totalcost(1,1);
    ColName = {'Index','Reduction','Abatement cost'};
    writecell([ColName;num2cell(linspace(1,step+1,step+1)'),num2cell([reduction,abatementcost./reduction])],sprintf("C:\\Users\\昊阳\\Desktop\\Alpha\\02 Figure\\FigureProject\\figs\\Figs@Scen%d@Vars%d@Pert%d%%@Mode%d@Gaps%d\\AbatementCost@Scen%d@Vars%d@Pert%d%%@Mode%d@Gaps%d.csv",s,v,p*100,m,g,s,v,p*100,m,g));
    % Metrics
    load("Params\Params.mat",'RowName','GeoParam','SLIB');
    ColName = {'城市','City','Participation','Variance','Correlation','Current state','Maximum','Rmax','Minimum','Rmin'};
    SLIB = sum(SLIB(:,1:16,1:6),3);
    weight = sum(SLIB,1)/sum(SLIB,"all");
    capacity = [];
    correlation = [];
    goal = (seq-1)/100;
    for i = seq
        amount = sum(solution(i).amount,3);
        capacity = [capacity,amount(:,end)];
        ColName = [ColName,cellstr(sprintf("R%d",i-1))];
    end
    for i = 1:337
        correlation = [correlation;corr(capacity(i,:)',goal')];
    end
    participation = mean(capacity,2);
    variance = var(capacity,0,2);
    [maximum,rmax] = max(fliplr(capacity),[],2);
    [minimum,rmin] = min(fliplr(capacity),[],2);
    rmax = (101 - rmax)/100;
    rmin = (101 - rmin)/100;
    city = participation ~= 0;
    participation = normalize(participation(city),'range');
    variance = normalize(variance(city),'range');
    capacity = capacity(city,:);
    correlation = correlation(city,:);
    current = GeoParam.Ref(:,3);
    current = current(city);
    maximum = maximum(city,:);
    minimum = minimum(city,:);
    rmax = rmax(city,:);
    rmin = rmin(city,:);
    RowName = RowName(city,:);
    writecell([ColName;RowName,num2cell([participation,variance,correlation,current,maximum,rmax,minimum,rmin,capacity])],sprintf("C:\\Users\\昊阳\\Desktop\\Alpha\\02 Figure\\FigureProject\\figs\\Figs@Scen%d@Vars%d@Pert%d%%@Mode%d@Gaps%d\\Metrics@Scen%d@Vars%d@Pert%d%%@Mode%d@Gaps%d.csv",s,v,p*100,m,g,s,v,p*100,m,g));
end
end