function [] = cobra(s,v,p,m,g)
tic;
%% Global settings
city = 337;
time = (2035-2020+1);
type = 6;
fcr = 7.0467;
%% Data import, specified for CHN
if exist('Params\Params.mat','file') == 0
    DT = importdata("Data/BatParam.xlsx");
    DT = DT.data;
    BatComp = DT.BatComp;
    MatComp = DT.MatComp;
    MatInt = zeros(16,10);
    % Supposing type4 Car 200 miles composition
    i = 4;
    Bat = BatComp(25*i-24:25*i-5,:);
    BatT = Bat;
    BatT(2,:) = BatT(2,:)*372/175;
    BatT = BatT./repmat(sum(BatT,1),20,1);
    Mat = Bat(2:20,:)'*MatComp(10:28,:);
    MatT = BatT(3:20,:)'*MatComp(11:28,:);
    for j = 1:8
        Mat(j,:) = Mat(j,:) + Bat(1,j)*MatComp(j,:);
        MatT(j,:) = MatT(j,:) + BatT(1,j)*MatComp(j,:) + BatT(2,j)*MatComp(9,:);
    end
    MatInt(1:8,:) = Mat;
    MatInt(9:16,:) = MatT;
    % ElePer in t/GWh, T means Titanium oxide anode instead of graphite
    DT = importdata("Data/SLIB.xls");
    PV = DT.data.PV;
    CV = DT.data.CV;
    SLIB = zeros(city,2060-2020+1,7);
    % Supposing all LTO batteries used LFP as cathode
    Den = zeros(1,7);
    Den(1:6) = BatComp(25*i-4,[1,2,3,4,6,8]);
    Den(7) = Den(1,1)/2;
    for i = 1:7
        SLIB(:,:,i) = (PV(city*i-city+2:city*i+1,:) + CV(city*i-city+2:city*i+1,:))/Den(i)*1e6;
    end
    MatInt = MatInt([1,2,3,4,6,8,9],:);
    writematrix(MatInt',"Data/MatInt.csv");
    DT = importdata("Data/PltParam.xlsx");
    PltParam = DT.data;
    DT = importdata("Data/GeoParam.xlsx");
    RowName = DT.textdata.Cost(2:end,2);
    GeoParam = DT.data;
    GeoParam.EF = GeoParam.EF(2:end,:);
    DT = importdata("Data/ScenParam.xlsx");
    ScenParam = DT.data(:,1:end-2);
    DT = importdata("Data/LogParam.xlsx");
    RowName = [DT.textdata(2:338,2),RowName];
    DT = DT.data;
    LogParam.Dist = reshape(DT(:,1),city,city)';
    LogParam.Toll = reshape(DT(:,2),city,city)';
    LogParam.Prov = reshape(DT(:,3),city,city)';
    save("Params/Params.mat",'GeoParam','PltParam','SLIB','ScenParam','LogParam','RowName');
else
    load("Params\Params.mat");
end
disp("Parameters successfully loaded!")
%% Data preprocessing, specified for CHN
fli = ScenParam(1,s);
fni = ScenParam(2,s);
fco = ScenParam(3,s);
ptp = ScenParam(4,s);
plc = ScenParam(5,s);
pns = ScenParam(6,s);
pcs = ScenParam(7,s);
pms = ScenParam(8,s);
fct = ScenParam(9,s);
fcp = ScenParam(10,s);
lr1 = ScenParam(11,s);
lr2 = ScenParam(12,s);
fgd = ScenParam(13,s);
SLIB = SLIB(:,1:time,1:type);
ALIB = sum(sum(SLIB,3),1);
if exist(sprintf("Params/RecParam@Scen%d@Vars%d@Pert%d%%.mat",s,v,p*100),'file') == 0
    %% 0 Precalculation
    % 0.1 Perturbations for sensitivity analysis
    fpt = ones(45,1);
    if v ~= 0
        fpt(v) = 1 + p;
    end
    % I Unit price
    GeoParam.Cost(:,1) = GeoParam.Cost(:,1)*fpt(1); % Labor price
    GeoParam.Cost(:,2) = GeoParam.Cost(:,2)*fpt(2); % Land price
    GeoParam.Cost(:,3) = GeoParam.Cost(:,3)*fpt(3); % Electricity price, energy charge
    GeoParam.Cost(:,4) = GeoParam.Cost(:,4)*fpt(4); % Electricity price, capacity charge
    GeoParam.Cost(:,5) = GeoParam.Cost(:,5)*fpt(5); % Gas price
    GeoParam.Cost(:,6) = GeoParam.Cost(:,6)*fpt(6); % Water price
    ptp = ptp*fpt(7); % NCM precursor price
    plc = plc*fpt(8); % Lithium carbonate price
    pns = pns*fpt(9); % Nickel sulfate price
    pcs = pcs*fpt(10); % Cobalt sulfate price
    pms = pms*fpt(11); % Manganese sulfate price
    PltParam.Input(3:13,2) = PltParam.Input(3:13,2)*fpt(12); % Base chemical price
    PltParam.Output(5:15,1) = PltParam.Output(5:15,1)*fpt(13); % Recycled byproduct price
    PltParam.Output(16:17,1) = PltParam.Output(16:17,1)*fpt(14); % Waste disposal price
    PltParam.EqpParam(:,4) = PltParam.EqpParam(:,4)*fpt(15); % Equipment price
    c0 = 260000*fpt(16); % Dedicated vehicle price
    l0 = (8155+6742)*12*fpt(17); % Driver and supervision price
    f0 = 90.05/12*fpt(18); % Diesel price per litre
    LogParam.Toll = LogParam.Toll*fpt(19); % Toll price per route
    i0 = 10410*fpt(20); % Insurance price per truck
    p0 = 37.5/0.4*fpt(21); % Package price per t of batteries
    e0 = 2800*fpt(22); % Equipment price per square meter for storage
    s0 = 80*fpt(23); % Storage insurance price
    % II Intensity assumption
    labor0 = 1060*fpt(24); % Labor consumption
    land0 = 500*fpt(25); % Land consumption
    PltParam.EqpParam(:,5) = PltParam.EqpParam(:,5)*fpt(26); % Electricity consumption, energy
    elec0 = fpt(27); % Electricity consumption, capacity
    gas0 = 2020000*fpt(28); % Gas consumption
    water01 = 81540*fpt(29); % Water consumption, part 1
    water02 = 60000*fpt(29); % Water consumption, part 2
    water03 = 819500*fpt(29); % Water consumption, part 3
    PltParam.Output(1,2:2+type-1) = PltParam.Output(1,2:2+type-1)*fpt(30); % NCM precursor yield*
    PltParam.Output(4,2:2+type-1) = PltParam.Output(4,2:2+type-1)*fpt(31); % Lithium carbonate yield*
    PltParam.Output(2,2:2+type-1) = PltParam.Output(2,2:2+type-1)*fpt(32); % Cobalt sulfate yield*
    PltParam.Output(3,2:2+type-1) = PltParam.Output(3,2:2+type-1)*fpt(33); % Manganese sulfate yield*
    PltParam.Output(5:15,2:2+type-1) = PltParam.Output(5:15,2:2+type-1)*fpt(34); % Recycled byproduct yield*
    PltParam.Output(16:17,2:2+type-1) = PltParam.Output(16:17,2:2+type-1)*fpt(35); % Waste yield*
    LogParam.Dist = LogParam.Dist*fpt(36); % Route distance
    dist0 = 120000*fpt(37); % Annual distance
    cap0 = 9.965/426*400*fpt(38); % Capacity per truck
    fuel0 = 0.259*fpt(39); % Fuel consumption per km
    lab0 = 0.005*fpt(40); % Labor consumption per square meter for storage
    elec01 = 12*fpt(41); % Electricty consumption per month per square meter for storage, energy part
    elec02 = 0.1*fpt(42); % Electricity consumption per month per square meter for storage, capacity part
    water0 = 0.025*fpt(43); % Water consumption per month per sqaure meter for storage
    stor0 = 2.0*0.7*fpt(44); % Storage capacity per square meter
    time0 = fpt(45); % Storage waiting time
    fprintf("A %d%% Perturbation of variable %d successfully applied!\n",p*100,v);
    % 0.2 Learning curve, using different learning rate on different cathodes, based on cumulative amount of retired batteries
    CLIB = cumsum(squeeze(sum(SLIB,1)));
    CLIB = [CLIB(:,1),sum(CLIB(:,2:type),2)];
    LR = [(CLIB./repmat(CLIB(4,:),time,1)).^log2(1-lr1),(CLIB./repmat(CLIB(4,:),time,1)).^log2(1-lr2)];
    flc1 = repmat(LR(:,2)',city,1,type);
    flc1(:,:,1) = repmat(LR(:,1)',city,1,1);
    flc2 = repmat(LR(:,4)',city,1,type);
    flc2(:,:,1) = repmat(LR(:,3)',city,1,1);
    %% 1 Fixed cost
    % 1.1 ISBL
    fer = 0.5;
    fp = 0.6;
    fi = 0.3;
    fel = 0.2;
    fc = 0.3;
    fs = 0.2;
    fl = 0.1;
    fis = fer + fp + fi + fel + fc + fs + fl;
    ISBL = zeros(city,time,type);
    for j = 1:type
        ISBL(:,:,j) = repmat(PltParam.EqpParam(1:37,4)'*PltParam.EqpParam(1:37,j+5),city,time)*fis/fcr;
    end
    ISBL = ISBL.*flc1;
    % 1.2 OSBL, altered
    % fos = 0.4;
    OSBL = zeros(city,time,type);
    for j = 1:type
        OSBL(:,:,j) = repmat(PltParam.EqpParam(38:50,4)'*PltParam.EqpParam(38:50,j+5),city,time)/fcr;
    end
    OSBL = OSBL.*flc1;
    % 1.3 Design and engineering
    fde = 0.25;
    DE = (ISBL + OSBL)*fde;
    % 1.4 Contingency
    fx = 0.1;
    X = (ISBL + OSBL)*fx;
    % 1.5 Land, altered
    Land = zeros(city,time,type);
    land = land0*2000/3*PltParam.EqpParam(48,6:6+type-1);
    for i = 1:type
        Land(:,:,i) = repmat(land(i)*GeoParam.Cost(:,2),1,time)/fcr;
    end
    % Land = Land.*flc1;
    % discount rate calculation
    fdr = 0.08;
    life = 20;
    fdp = fdr*(1 + fdr)^life/((1 + fdr)^life-1);
    fdl = fdr*(1 + fdr)^life/((1 + fdr)^life-1);
    PT = (ISBL + OSBL + DE + X)*fdp;
    LD = Land*fdl;
    CI = PT/fdp + LD/fdl;
    %% 2 Variable cost of production
    % 2.1 Batteries, altered
    BT = zeros(city,time,type);
    battery = sum(PltParam.MatInt(1:3,1:type).*repmat([1/0.1872;1/0.220;1/0.205],1,type).*repmat([fli*plc;fni*pns;fco*pcs],1,type),1);
    for j = 1:type
        BT(:,:,j) = repmat(battery(j),city,time)/fcr;
    end
    % 2.2 Raw materials, altered
    RM = zeros(city,time,type);
    rm = PltParam.Input(:,3:3+type-1);
    pin = PltParam.Input(:,2);
    pin(1) = pcs/PltParam.Input(1,1);
    pin(2) = pms/PltParam.Input(2,1);
    for j = 1:type
        RM(:,:,j) = repmat(rm(:,j)'*pin,city,time)/fcr;
    end
    % 2.3 Utilities
    Elec = zeros(city,time,type);
    elec = sum(PltParam.EqpParam(:,6:6+type-1).*repmat(PltParam.EqpParam(:,5),1,type),1);
    for j = 1:type
        Elec(:,:,j) = (repmat(elec(j),city,time).*repmat(GeoParam.Cost(:,3),1,time) + 12*elec(j)*elec0/7200*repmat(GeoParam.Cost(:,4),1,time))/fcr;
    end
    % Elec = Elec.*flc2;
    Gas = zeros(city,time,type);
    gas = gas0*PltParam.EqpParam(38,6:6+type-1);
    for j = 1:type
        Gas(:,:,j) = repmat(gas(j)*GeoParam.Cost(:,5),1,time)/fcr;
    end
    % Gas = Gas.*flc2;
    Water = zeros(city,time,type);
    water_t = water01*PltParam.EqpParam(48,6:6+type-1);
    water_r = water02*PltParam.EqpParam(44,6:6+type-1) + water03*PltParam.EqpParam(45,6:6+type-1);
    for j = 1:type
        Water(:,:,j) = (repmat(water_t(j)*GeoParam.Cost(:,6) + water_r(j)*GeoParam.Cost(:,6)*0.6,1,time))/fcr;
    end
    % Water = Water.*flc2;
    UT = Elec + Gas + Water;
    % 2.4 Consumables, deleted
    % 2.5 Effluent disposal, altered
    SW = zeros(city,time,type);
    for j = 1:type
        SW(:,:,j) = repmat(PltParam.Output(16:17,1)'*PltParam.Output(16:17,j+1),city,time)/fcr;
    end
    % SW = SW.*flc2;
    WW = Water*0.3;
    WD = SW + WW;
    % 2.6 Packaging and shipping, BatLog V3.2
    % PS = 0.02*(Bat + RM);
    RAA = LogParam.Dist/dist0/cap0;
    cap = c0*0.08*(1+0.08)^10/((1+0.08)^10-1)*RAA;
    labor = l0*RAA;
    fuel = dist0*fuel0*f0*RAA;
    toll = 2.1*LogParam.Toll/cap0;
    maint = 0.13*dist0*RAA;
    ins = i0*RAA;
    pack = p0;
    stor = repmat(((e0*ones(city,1) + GeoParam.Cost(:,2))*(0.05*(1+0.05)^20/((1+0.05)^20-1))*(1+0.01) ...
        + lab0*GeoParam.Cost(:,1) + (elec01*GeoParam.Cost(:,3)+elec02*GeoParam.Cost(:,4)+water0*GeoParam.Cost(:,6))*12 ...
        + s0*stor0*ones(city,1))/stor0/12,1,city);
    stor = time0*(stor.*LogParam.Prov);
    trans = cap + labor + fuel + toll + maint + ins + pack;
    SC = zeros(city,city,time,type);
    TC = zeros(city,city,time,type);
    for n = 1:city
        SC(n,:,:,:) = repmat(stor(n,:)',1,time,type)/fcr.*flc1;
        TC(n,:,:,:) = repmat(trans(n,:)',1,time,type)/fcr.*flc1;
    end
    PS = SC + TC;
    %% 3 FCOP
    % 3.1 Labor
    fsm = 0.25;
    foh = 0.5;
    LB = zeros(city,time,type);
    lb = labor0*PltParam.EqpParam(48,6:12);
    for i = 1:type
        LB(:,:,i) = repmat(lb(i)*GeoParam.Cost(:,1)*(1+fsm)*(1+foh),1,time)/fcr;
    end
    LB = LB.*flc2;
    % 3.2 Maintenance
    fmt = 0.05;
    MT = fmt*ISBL;
    % 3.3 Land, rent and local property taxes, deleted
    % 3.4 Insurance
    fins = 0.01;
    IS = fins*(ISBL + OSBL);
    % 3.5 Interest payments, deleted
    % 3.6 Corporate overhead charges, altered
    RV = zeros(city,time,type);
    rv = PltParam.Output(1:15,2:2+type-1);
    DF = zeros(city,time,type);
    pout = PltParam.Output(1:15,1);
    pout(1) = ptp;
    pout(2) = pcs;
    pout(3) = pms;
    pout(4) = plc;
    for j = 1:type
        RV(:,:,j) = repmat(rv(:,j)'*pout,city,time)/fcr;
        DF(:,:,j) = repmat((rv(1:4,j)'*pout(1:4))/(rv(:,j)'*pout),city,time);
    end
    frd = 0.01;
    fgsa = 0.65;
    RD = frd*RV;
    GA = fgsa*LB;
    OH = RD + GA;
    % 3.7 License fees and royalities, deleted
    % 3.8 Taxes, altered, go to part5
    % 3.9 Carbon Tax, go to part5
    %% 4 Carbon emission
    % 4.0 Grid decarbonization scenarios
    if fgd ~= 1
        gdd = zeros(1,time);
        for i = 1:time
            gdd(i) = (i - 1)/(time-1)*fgd;
        end
        GeoParam.EF = GeoParam.EF.*repmat(gdd,city,1);
    end
    % 4.1 Process emissions
    PC = zeros(city,time,type);
    pc = sum(PltParam.EqpParam(1:37,6:6+type-1).*repmat(PltParam.EqpParam(1:37,5),1,type),1);
    for i = 1:type
        PC(:,:,i) = pc(i)*GeoParam.EF/1e3 + repmat(35.8*gas(i)*PltParam.GWP(14,4)/1e6 + 165*PltParam.Input(8,2 + i)/1e3,city,time);
    end
    % 4.2 Upstream material emissions
    UM = zeros(city,time,type);
    fgwp = repmat(PltParam.GWP(18,:)'/1000,1,time);
    for i = 1:city
        fgwp(6,:) = GeoParam.EF(i,1:time)/3.6;
        me = PltParam.GWP(1:13,:)*fgwp;
        for j = 1:type
            UM(i,:,j) = rm(:,j)'*me;
        end
    end
    % 4.3 Waste treatment emissions
    WT = zeros(city,time,type);
    wt = sum(PltParam.EqpParam(38:50,6:6+type-1).*repmat(PltParam.EqpParam(38:50,5),1,type),1);
    for i = 1:type
        WT(:,:,i) = wt(i)*GeoParam.EF/1e3;
    end
    % 4.4 Logistics emissions
    TE = zeros(city,city,time,type);
    SE = zeros(city,city,time,type);
    ef = zeros(city,city,time,type);
    for n = 1:city
        dc = 35.8*fuel0/cap0*LogParam.Dist(n,:);
        TE(n,:,:,:) = repmat(dc*PltParam.GWP(14,2)/1e6,1,1,time,type);
        ef(n,:,:,:) = repmat(GeoParam.EF(n,:),city,1,type);
        SE(n,:,:,:) = repmat(elec01/stor0*LogParam.Prov(n,:),1,1,time,type).*ef(n,:,:,:)/1e3;
    end
    LG = TE + SE;
    % 4.5 Primary source extraction emissions
    PM = zeros(city,time,type);
    fgwp(6,:) = mean(GeoParam.EF(:,1:time)/3.6,1);
    pe = PltParam.GWP(14:17,:)*fgwp;
    for j = 1:type
        PM(:,:,j) = repmat(PltParam.Output(1:4,j+1)'*pe,city,1);
    end
    filepath = sprintf("Params/RecParam@Scen%d@Vars%d@Pert%d%%.mat",s,v,p*100);
    save(filepath,'CI','PT','LD','BT','RM','UT','Elec','Gas','Water','WD','PS','SC','TC','LB','MT','IS','OH','RD','GA','RV','PC','UM','WT','LG','TE','SE','PM','DF');
else
    load(sprintf("Params/RecParam@Scen%d@Vars%d@Pert%d%%.mat",s,v,p*100));
end
disp("Parameters successfully calculated!")
%% 5 Optimization
fprintf("Optimizing under scenario %d, a %d%% perturbation of variable %d, mode %d, %d gaps ...\n\n",s,p*100,v,m,g);
bat = zeros(city,city,time,type);
tc = zeros(city,city,time,type);
te = zeros(city,city,time,type);
cpd = zeros(1,time);
if fcp ~= 0
    for i = 1:time
        cpd(i) = 7.10*exp(fcp*(i - 1));
    end
end
for i = 1:time
    for j = 1:type
        bat(:,:,i,j) = repmat(SLIB(:,i,j),1,city);
    end
end
od = sdpvar(city,city,time,'full');
capabilities = binvar(city,time);
flow = repmat(od,1,1,1,type).*bat; % Faster calculation
amount = squeeze(sum(flow,1));
C = [];
% upper and lower bound settings
ub = 0.10;
lb = 0.01;
switch m
    case 0 % Proportional established facilites
        Ref = zeros(city,time);
        for i = 1:city
            if GeoParam.Ref(i,2) == 1
                Ref(i,:) = ones(1,time);
            end
        end
        Prop = GeoParam.Ref(:,3)/sum(GeoParam.Ref(:,3),"all");
        C = [C,capabilities == Ref,sum(amount,3) <= repmat(sum(sum(amount,3),1),city,1).*repmat(Prop,1,time)];
    case 1 % Optimal established facilities
        Ref = zeros(city,time);
        for i = 1:city
            if GeoParam.Ref(i,1) == 1
                Ref(i,:) = ones(1,time);
            end
        end
        Ref = Ref + repmat(GeoParam.Ref(:,2),1,time);
        C = [C,capabilities <= Ref,capabilities(:,4) == GeoParam.Ref(:,2),sum(amount,3) >= capabilities.*repmat(ALIB,city,1)*lb,sum(amount,3) <= capabilities.*repmat(ALIB,city,1)*ub];
    case 2 % Free of established facilities
        Ref = zeros(city,time);
        for i = 1:city
            if GeoParam.Ref(i,1) == 1
                Ref(i,:) = ones(1,time);
            end
        end
        C = [C,capabilities <= Ref,sum(amount,3) >= capabilities.*repmat(ALIB,city,1)*lb,sum(amount,3) <= capabilities.*repmat(ALIB,city,1)*ub];
    case 3 % Intra-provincial recycling
        noc = [1,1,11,11,12,14,9,13,1,13,11,16,9,11,16,17,13,14,21,14,4,1,21,9,16,7,10,14,8,5,14];
        ex = [];
        for i = 1:31
            ex = blkdiag(ex,ones(noc(i)));
        end
        Ref = zeros(city,time);
        for i = 1:city
            if GeoParam.Ref(i,1) == 1 || GeoParam.Ref(i,1) == 2
                Ref(i,:) = ones(1,time);
            end
        end
        C = [C,capabilities <= Ref,od <= repmat(ex,1,1,time),sum(capabilities,1) == 31];
    case 4 % Fully local recycling
        C = [C,capabilities == ones(city,time),od == repmat(eye(city,city),1,1,time)];
end
% Associated constraints
C = [C,od >= zeros(city,city,time),squeeze(sum(od,2)) == ones(city,time),squeeze(sum(od,1)) <= repmat(capabilities*city,1,1)];
% Capacity constraints (very necessary)
for i = 1:time - 1
    C = [C,squeeze(sum(amount(:,i,:),3)) <= squeeze(sum(amount(:,i + 1,:),3))];
end
% Tax calculation
for n = 1:city
    EMISSION = PC + UM + WT + squeeze(LG(n,:,:,:));
    CT = EMISSION.*repmat(cpd,city,1,type);
    TX = max(fct*(RV/1.13 - BT/1.13 - RM/1.13 - Elec/1.13 - Gas/1.09 - Water/1.09 - WD - squeeze(PS(n,:,:,:)) - LB - MT - IS - RD - GA - PT - LD - CT),0) + max((RV - BT - RM - Elec)/1.13*0.13 - (Gas + Water)/1.09*0.09,0);
    COST = PT + LD + RM + UT + WD + squeeze(PS(n,:,:,:)) + LB + MT + IS + RD + GA + TX + CT;
    tc(n,:,:,:) = COST;
    te(n,:,:,:) = EMISSION;
end
cost = squeeze(sum(flow.*tc,1));
emission = squeeze(sum(flow.*te,1));
Ops = sdpsettings('solver','gurobi','verbose',0,'gurobi.TuneTimeLimit',Inf,'gurobi.MIPGap',1e-5);
% Ops = sdpsettings('solver','gurobi','verbose',0,'gurobi.TuneTimeLimit',Inf,'gurobi.MIPGap',1e-5,'usex0',1);
solution(g + 1) = struct('od',[],'capabilities',[],'flow',[],'amount',[],'cost',[],'emission',[],'profit',[],'reduction',[]);
% Warm start
% if exist(sprintf("Solutions/Sol@Scen%d@Vars0@Pert0%%@Mode%dGaps%d.mat",s,m,g),'file') ~= 0
%     solution0 = load(sprintf("Solutions/Sol@Scen%d@Vars0@Pert0%%@Mode%dGaps%d.mat",s,m,g));
%     solution0 = solution0.solution;
% end
% Pareto frontier search, approximately...
for i = [1, g + 1, 2:g]
    if i == 1 || i == g+1
        switch i
            case 1
                Obj = sum(cost,"all");
            case g + 1
                Obj = sum(emission,"all");
        end
        % Warm start
        % if exist('solution0','var') ~= 0
        %     assign(od,solution0(i).od);
        %     assign(capabilities,solution0(i).capabilities);
        % end
        sol = optimize(C,Obj,Ops);
    else
        lb = sum(solution(g + 1).emission,"all");
        ub = sum(solution(1).emission,"all");
        Cp = [C,sum(emission,"all") <= lb + (g + 1 - i)/g*(ub - lb)];
        Obj = sum(cost,"all");
        % Warm start
        % if exist('solution0','var') ~= 0
        %     assign(od,solution0(i).od);
        %     assign(capabilities,solution0(i).capabilities);
        % end
        sol = optimize(Cp,Obj,Ops);
    end
    if sol.problem == 0
        solution(i).od = value(od);
        solution(i).capabilities = value(capabilities);
        solution(i).flow = value(flow);
        solution(i).amount = value(amount);
        solution(i).cost = value(cost);
        solution(i).emission = value(emission);
        solution(i).profit = value(amount.*RV - cost - amount.*BT);
        solution(i).reduction = value(amount.*PM - emission.*DF);
        fprintf("Optimal solution %d found!\n" + ...
            "Number of selected cities: %d\n" + ...
            "Cost: %.2e $\n" + ...
            "Emission: %.2e t CO2e\n" + ...
            "Abatement percentage: %.1f %%\n" + ...
            "Abatement cost: %.0f $/t CO2e\n\n", ...
            i,sum(solution(i).capabilities(:,time),"all"),sum(solution(i).cost,"all"),sum(solution(i).emission,"all"), ...
            (sum(solution(1).emission,"all")-sum(solution(i).emission,"all"))/(sum(solution(1).emission,"all")-sum(solution(g+1).emission,"all"))*100, ...
            (sum(solution(i).cost,"all") - sum(solution(1).cost,"all"))/(sum(solution(1).emission,"all")-sum(solution(i).emission,"all")));
    else
        disp(sol.info)
    end
end
filepath = sprintf("Solutions/Sol@Scen%d@Vars%d@Pert%d%%@Mode%d@Gaps%d.mat",s,v,p*100,m,g);
save(filepath,'solution','-v7.3');
t = toc;
fprintf('Elapsed time is %.3f seconds.\n\n', t);
end