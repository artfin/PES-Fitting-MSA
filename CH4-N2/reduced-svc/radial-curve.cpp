#include <cassert>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <memory>
#include <thread>

#include <torch/script.h>

#include "ai_pes_ch4_n2_opt1.hpp"

const double HTOCM     = 2.194746313702e5; 

const double a0 = 2.0;
const double NN_BOND_LENGTH = 2.078; // a0 

extern "C" {
    void c_evmono(double* x, double* mono);
    void c_evpoly(double* mono, double* poly);
}

#define ATOMX(x, i) x[3*i]
#define ATOMY(x, i) x[3*i + 1]
#define ATOMZ(x, i) x[3*i + 2]

template <typename T>
std::vector<T> linspace(const T start, const T end, const size_t size) {
    
    const T step = (end - start) / (size - 1);
    
    std::vector<T> v(size);
    for (size_t k = 0; k < size; ++k) {
        v[k] = start + step * k;
    }

    return v;
}


struct StandardScaler
{
    StandardScaler() = default;

    void transform(double *x, size_t sz) {

        assert(sz == mean.size());
        assert(sz == std.size());

        const double EPS = 1e-9;
        for (size_t k = 0; k < sz; ++k) {
            x[k] = (x[k] - mean[k]) / (std[k] + EPS);
        }
    }

public:
    std::vector<double> mean;
    std::vector<double> std;
};

struct NNPIP
{
public:
    NNPIP(const size_t NATOMS, std::string const& pt_fname);
    ~NNPIP();

    double pes(std::vector<double> const& x);
    //   * input: cartesian coordinates of atoms (7 atoms -> 21 coordinates)
    //            order: [H H H H N N C]
    //            units: Bohr 
    //   * output: energy / Hartree 
    
    const size_t NATOMS;
    const size_t NMON = 2892;
    const size_t NPOLY = 650;

    const size_t NDIS;

    double *yij;
    double *mono;
    double *poly;

    StandardScaler xscaler;
    StandardScaler yscaler;

    torch::jit::script::Module model;
    at::Tensor t;
};

NNPIP::NNPIP(const size_t NATOMS, std::string const& pt_fname)
    : NATOMS(NATOMS), NDIS(NATOMS * (NATOMS - 1) / 2)
{
    yij = new double [NDIS];
    mono = new double [NMON];
    poly = new double [NPOLY];

    try {
        model = torch::jit::load(pt_fname);
    } catch (const c10::Error& e) {
        std::cerr << ": ERROR: could not load the model\n";
        exit(1);
    }

    // analogous to py:with torch.no_grad()
    torch::NoGradGuard no_grad;

    xscaler.mean = {1.0000000000000000, 0.0578080064001517, 0.1251803226500610, 1.4227865917870646, 0.2384248127208684, 1.1093474478617720, 0.0012355425846187, 0.0072364248929278, 0.0822484564040766, 0.1781048846220852, 0.7591206321633946, 0.0102212659073472, 0.0190066091358915, 0.0298460949841787, 0.2544207200164452, 0.0143897627579144, 0.0250392221012288, 0.0848069066721484, 0.0067427057495548, 0.0641291643659852, 0.1388684714543582, 0.7891823372254639, 0.1322479787494081, 0.1025543133397855, 0.7891823372254639, 0.1322479787494081, 0.4102172533591420, 0.0047895609458967, 0.0156701131787734, 0.5060804214422631, 0.0258455772319981, 0.2051086266795706, 0.0001546656193905, 0.0017579134229774, 0.0102958883101318, 0.0438832503625887, 0.0950269656645319, 0.1800111094984997, 0.0007706345574044, 0.0012795013641738, 0.0023792534641145, 0.0109070100630478, 0.0202817614754129, 0.0318484678205194, 0.0904965972780522, 0.0021996035088877, 0.0018013151448936, 0.0102367807554787, 0.0010215909284662, 0.0031019133016845, 0.0031344179015384, 0.0178127347372034, 0.0017925886948756, 0.0016073646467486, 0.0036356700210159, 0.0067605871584710, 0.0106161559401731, 0.0904965972780523, 0.0102367807554787, 0.0178127347372034, 0.0010306595106333, 0.0008440540812637, 0.0071950734996240, 0.0024856086575113, 0.0023983578332080, 0.0013706460129713, 0.0080277094866128, 0.0456210576012163, 0.0987900996036128, 0.1403547559849461, 0.0056694676241161, 0.0105424666687037, 0.0165548446496693, 0.0470401627388996, 0.0026605410984714, 0.0046295328557404, 0.0470401627388996, 0.0037400017074757, 0.0059284604019095, 0.0128377820330298, 0.0456210576012163, 0.0987900996036128, 0.5614190239397843, 0.0940803254777993, 0.1459129019497762, 0.1403547559849461, 0.0056694676241161, 0.0105424666687037, 0.0165548446496693, 0.0940803254777993, 0.0106421643938858, 0.0185181314229615, 0.0244514929517556, 0.0470401627388995, 0.0026605410984714, 0.0046295328557404, 0.0470401627388996, 0.0037400017074757, 0.0237138416076379, 0.0513511281321192, 0.1459129019497762, 0.0244514929517556, 0.2918258038995524, 0.0489029859035112, 0.0758455771804717, 0.0252818590601572, 0.1459129019497761, 0.0244514929517556, 0.0252818590601572, 0.0001929712888249, 0.0005995587845595, 0.0009058580029296, 0.0068145230943688, 0.0222952269225445, 0.0292555002417258, 0.0633513104430212, 0.5400333284954995, 0.0008091785248974, 0.0024218648884578, 0.0037361437999639, 0.0904965972780522, 0.0301655324260174, 0.0010543518258373, 0.0032007878673016, 0.0032353576969786, 0.0275795555570130, 0.0027495555213042, 0.0072564513949084, 0.0091931851856710, 0.0016728184534647, 0.0053132872117089, 0.0173836000625773, 0.2807095119698922, 0.0143358625704157, 0.2807095119698917, 0.0143358625704157, 0.0118569208038189, 0.0256755640660595, 0.1459129019497763, 0.0244514929517556, 0.0379227885902358, 0.1459129019497754, 0.0244514929517555, 0.1516911543609432, 0.0006282087668784, 0.0019615898236818, 0.1800111094984991, 0.0053702237219927, 0.0379227885902353, 0.0002200561694792, 0.0009379258679005, 0.0054933194393223, 0.0104060833699877, 0.0225338487676177, 0.0160073370604487, 0.0000964682825412, 0.0008223363865821, 0.0013653430388398, 0.0025388774454038, 0.0038795869185478, 0.0072141545712602, 0.0113283932459992, 0.0107297787674639, 0.0000655166919488, 0.0002753470769448, 0.0015647831898466, 0.0012814435178688, 0.0018205943002449, 0.0001180130943100, 0.0001278830820418, 0.0003882985079374, 0.0007267529376565, 0.0022066803272613, 0.0022298038816831, 0.0031679650183941, 0.0001090278558566, 0.0002085184298828, 0.0002243968312034, 0.0006376177899145, 0.0000209626821014, 0.0000613616941269, 0.0002547243630397, 0.0002012104250963, 0.0005717342168766, 0.0000501401146951, 0.0000447697166760, 0.0002741121288607, 0.0004551143462799, 0.0008462924818013, 0.0038795869185478, 0.0072141545712602, 0.0113283932459992, 0.0321893363023916, 0.0007823915949233, 0.0007823915949233, 0.0012814435178688, 0.0072823772009795, 0.0006376177899145, 0.0018205943002449, 0.0007267529376565, 0.0022066803272613, 0.0022298038816831, 0.0126718600735763, 0.0012752355798291, 0.0017152026506298, 0.0031679650183941, 0.0000304402085715, 0.0001290182900834, 0.0010998063993201, 0.0009006816221738, 0.0025592635255469, 0.0001501758953763, 0.0002912394009691, 0.0003111492937290, 0.0017682453351685, 0.0001158538979457, 0.0002095397385441, 0.0008841226675842, 0.0000379521609415, 0.0003666021331067, 0.0003002272073913, 0.0025592635255469, 0.0008841226675842, 0.0001715779101428, 0.0009750683846710, 0.0057108587101573, 0.0081136286322695, 0.0175696536396661, 0.0004274507397453, 0.0007097057864409, 0.0013197093791158, 0.0020166106295408, 0.0037499200551484, 0.0058885027491668, 0.0004066874231487, 0.0003330473931304, 0.0001888832148754, 0.0005735166007854, 0.0005795264166006, 0.0020166106295408, 0.0037499200551484, 0.0058885027491668, 0.0167320282050970, 0.0018926911009018, 0.0032934186366925, 0.0005716797488677, 0.0004681746204536, 0.0013303060706643, 0.0004595672700989, 0.0013303060706643, 0.0001267102213676, 0.0007421265859291, 0.0009750683846710, 0.0057108587101573, 0.0324545145290780, 0.0702786145586645, 0.0998474324544633, 0.0040332212590816, 0.0074998401102968, 0.0117770054983337, 0.0334640564101939, 0.0018926911009018, 0.0032934186366925, 0.0334640564101939, 0.0026606121413286, 0.0084349339697774, 0.0182654241448797, 0.0519007301157206, 0.0081136286322695, 0.0175696536396661, 0.0998474324544632, 0.0167320282050970, 0.0259503650578603, 0.0004274507397453, 0.0007097057864409, 0.0013197093791158, 0.0040332212590816, 0.0074998401102969, 0.0117770054983337, 0.0167320282050970, 0.0008133748462975, 0.0008133748462975, 0.0013321895725216, 0.0037853822018036, 0.0003314339489544, 0.0037853822018036, 0.0007555328595018, 0.0022940664031415, 0.0023181056664026, 0.0065868372733851, 0.0006628678979087, 0.0008915629343269, 0.0065868372733851, 0.0009191345401978, 0.0010482349065913, 0.0019492097488490, 0.0030608457769765, 0.0173946281604669, 0.0009838214925069, 0.0017119201527693, 0.0020166106295408, 0.0037499200551484, 0.0058885027491668, 0.0334640564101939, 0.0037853822018036, 0.0065868372733851, 0.0086973140802335, 0.0004066874231487, 0.0003330473931304, 0.0018926911009018, 0.0006628678979087, 0.0004919107462534, 0.0001888832148754, 0.0005735166007854, 0.0005795264166006, 0.0032934186366925, 0.0003314339489544, 0.0008915629343269, 0.0008559600763846, 0.0020166106295408, 0.0037499200551484, 0.0058885027491668, 0.0334640564101940, 0.0037853822018036, 0.0065868372733851, 0.0086973140802335, 0.0167320282050970, 0.0018926911009018, 0.0032934186366925, 0.0005716797488677, 0.0004681746204536, 0.0026606121413286, 0.0009191345401978, 0.0006914935581978, 0.0013303060706643, 0.0004595672700989, 0.0013303060706643, 0.0005068408854705, 0.0029685063437166, 0.0084349339697774, 0.0182654241448797, 0.0010482349065913, 0.0019492097488490, 0.0030608457769765, 0.0086973140802335, 0.0006914935581978, 0.0168698679395547, 0.0365308482897594, 0.1038014602314410, 0.0173946281604669, 0.0519007301157206, 0.0020964698131827, 0.0038984194976981, 0.0061216915539529, 0.0173946281604669, 0.0019676429850138, 0.0034238403055385, 0.0173946281604669, 0.0009838214925069, 0.0017119201527693, 0.0173946281604669, 0.0013829871163956, 0.0043844816110719, 0.0094943738230315, 0.0539560351293630, 0.0090417337674801, 0.0035057957281983, 0.0014614938703573, 0.0031647912743438, 0.0089926725215605, 0.0015069556279133, 0.0084349339697774, 0.0182654241448797, 0.0519007301157206, 0.0086973140802335, 0.1038014602314412, 0.0173946281604669, 0.0539560351293631, 0.0269780175646815, 0.0010482349065913, 0.0019492097488490, 0.0030608457769765, 0.0086973140802335, 0.0009838214925069, 0.0017119201527693, 0.0173946281604669, 0.0019676429850138, 0.0034238403055385, 0.0090417337674801, 0.0045208668837400, 0.0086973140802334, 0.0006914935581978, 0.0014614938703573, 0.0031647912743438, 0.0269780175646815, 0.0045208668837400, 0.0140231829127932, 0.0089926725215605, 0.0015069556279133, 0.0000241562081973, 0.0000193610921382, 0.0002745569623400, 0.0008530441996594, 0.0012888426206313, 0.0036358545330342, 0.0118955062223424, 0.0006252839119337, 0.0036622129595482, 0.0312182501099630, 0.0676015463028530, 0.1920880447253838, 0.0000471124172160, 0.0000918110889977, 0.0001012932288282, 0.0003031698281520, 0.0001601683935985, 0.0002978357163041, 0.0008634662666895, 0.0025843476678132, 0.0039868014776827, 0.0038795869185478, 0.0072141545712602, 0.0113283932459992, 0.0643786726047833, 0.0002639407125164, 0.0002254892110322, 0.0036411886004897, 0.0000637374834766, 0.0005005344058691, 0.0003923674442347, 0.0063359300367882, 0.0002878220888965, 0.0008614492226044, 0.0013289338258942, 0.0321893363023917, 0.0012931956395159, 0.0024047181904201, 0.0037761310819997, 0.0321893363023917, 0.0036411886004897, 0.0063359300367881, 0.0001250521974560, 0.0001056589622267, 0.0025592635255469, 0.0008530878418490, 0.0001217637467985, 0.0001319841017450, 0.0004006756579632, 0.0011250882306207, 0.0034155285455636, 0.0034524176631722, 0.0098099554634911, 0.0001675716218545, 0.0003181887778669, 0.0003441902473011, 0.0019560153645429, 0.0001292511324281, 0.0009780076822714, 0.0002771724384575, 0.0011453706607193, 0.0009083649269091, 0.0051621908743151, 0.0004534028776243, 0.0007940665121109, 0.0025810954371576, 0.0003000998519728, 0.0003750294102069, 0.0011385095151879, 0.0011508058877241, 0.0098099554634911, 0.0009780076822714, 0.0025810954371576, 0.0001002116933123, 0.0001986341111381, 0.0002094039537397, 0.0017850477495626, 0.0002318550576209, 0.0004211911748782, 0.0005950159165209, 0.0002140722067685, 0.0006651190074942, 0.0010049112636751, 0.0037798369015665, 0.0123665765430119, 0.0162272572645390, 0.0351393072793323, 0.0998474324544632, 0.0004488300657297, 0.0013433448165383, 0.0020723407946672, 0.0167320282050970, 0.0167320282050970, 0.0005848212535705, 0.0017753929258690, 0.0017945679020316, 0.0050992182617224, 0.0005083687333855, 0.0013416543059124, 0.0050992182617224, 0.0009278684410436, 0.0004911901340055, 0.0016070376970058, 0.0037798369015665, 0.0123665765430119, 0.1996948649089268, 0.0101984365234448, 0.0162272572645389, 0.0351393072793322, 0.1996948649089263, 0.0334640564101939, 0.0519007301157205, 0.0998474324544633, 0.0004488300657297, 0.0013433448165383, 0.0020723407946672, 0.0334640564101940, 0.0010167374667710, 0.0026833086118249, 0.0167320282050969, 0.0167320282050969, 0.0005848212535705, 0.0017753929258690, 0.0017945679020316, 0.0101984365234448, 0.0010167374667710, 0.0026833086118249, 0.0026505754258980, 0.0050992182617224, 0.0005083687333855, 0.0013416543059124, 0.0050992182617224, 0.0009278684410436, 0.0019647605360220, 0.0064281507880233, 0.0519007301157206, 0.0026505754258980, 0.1038014602314412, 0.0053011508517959, 0.0519007301157204, 0.0026505754258980, 0.0002534204427353, 0.0014842531718583, 0.0084349339697774, 0.0182654241448797, 0.0259503650578603, 0.0010482349065913, 0.0019492097488490, 0.0030608457769765, 0.0086973140802334, 0.0004919107462534, 0.0008559600763846, 0.0086973140802334, 0.0006914935581978, 0.0021922408055360, 0.0047471869115158, 0.0269780175646815, 0.0045208668837400, 0.0084349339697773, 0.0182654241448796, 0.1038014602314408, 0.0173946281604669, 0.0269780175646814, 0.0259503650578602, 0.0010482349065913, 0.0019492097488490, 0.0030608457769764, 0.0173946281604670, 0.0019676429850138, 0.0034238403055385, 0.0045208668837400, 0.0086973140802333, 0.0004919107462534, 0.0008559600763846, 0.0086973140802334, 0.0006914935581978, 0.0087689632221438, 0.0189887476460631, 0.0539560351293631, 0.0090417337674801, 0.0539560351293631, 0.0090417337674801, 0.0280463658255864, 0.0539560351293631, 0.0090417337674801, 0.0140231829127932, 0.0140231829127932, 0.0539560351293629, 0.0090417337674801, 0.0140231829127932, 0.0000057755625272, 0.0000237335323501, 0.0000786393761294, 0.0000750529620988, 0.0001133955970819, 0.0008938070103577, 0.0027909236997204, 0.0024239030220228, 0.0079303374815616, 0.0960440223626922, 0.0104060833699876, 0.0225338487676176, 0.1920880447253838, 0.0001005213362686, 0.0003986700584723, 0.0004676916863465, 0.0321893363023916, 0.0107297787674638, 0.0000657513733618, 0.0005158908249215, 0.0004050031203961, 0.0098099554634911, 0.0000965289982168, 0.0005520242213603, 0.0032699851544970, 0.0000907348684736, 0.0002048279173127, 0.0008514504611400, 0.0006722463382221, 0.0057305117299110, 0.0005001666644347, 0.0017278969424269, 0.0019101705766370, 0.0003713846857058, 0.0006969017922610, 0.0021760846646530, 0.0998474324544634, 0.0029787219902197, 0.0998474324544625, 0.0029787219902197, 0.0009823802680110, 0.0032140753940116, 0.0519007301157210, 0.0026505754258980, 0.0035057957281983, 0.0519007301157198, 0.0026505754258980, 0.0140231829127933, 0.0021922408055359, 0.0047471869115157, 0.0269780175646816, 0.0045208668837400, 0.0070115914563966, 0.0269780175646808, 0.0045208668837400, 0.0280463658255862, 0.0001055627146990, 0.0002455524470356, 0.0640293482417938, 0.0016403124096754, 0.0070115914563963};
    xscaler.std  = {0.0000000000000000, 0.0626009625419131, 0.0000000000000000, 0.0000000000000000, 0.2476743166507446, 0.0000000000000000, 0.0020613095470546, 0.0078364086892011, 0.0890678101375990, 0.0000000000000001, 0.0000000000000000, 0.0167530852265173, 0.0367099747596739, 0.0310039508704734, 0.2642907726455265, 0.0232256073553260, 0.0461665504903757, 0.0880969242151756, 0.0110079566465079, 0.0694462180295620, 0.0000000000000000, 0.0000000000000000, 0.1373784355387044, 0.0000000000000000, 0.0000000000000000, 0.1373784355387044, 0.0000000000000000, 0.0097043261331695, 0.0000000000000000, 0.0000000000000000, 0.0495065119284475, 0.0000000000000000, 0.0002580353941819, 0.0029328035850720, 0.0111495372107590, 0.0475216822588543, 0.0000000000000000, 0.0000000000000000, 0.0017055028826631, 0.0020971566140394, 0.0045953664848916, 0.0178770487735160, 0.0391728449046789, 0.0330840041932008, 0.0940073419132751, 0.0048234521650634, 0.0029073890224833, 0.0165225413656344, 0.0021993503987540, 0.0090612985643702, 0.0057791436860256, 0.0328425745133837, 0.0038270311306766, 0.0045099043797281, 0.0059590162578387, 0.0130576149682264, 0.0110280013977337, 0.0940073419132751, 0.0165225413656344, 0.0328425745133837, 0.0023276417176857, 0.0013779795647277, 0.0117464798397185, 0.0053592165956885, 0.0039154932799062, 0.0022867084852781, 0.0086932999797671, 0.0494035739313908, 0.0000000000000000, 0.0000000000000000, 0.0092924961699238, 0.0203620584053571, 0.0171970768858957, 0.0488650490212885, 0.0042942113741118, 0.0085357908271799, 0.0488650490212885, 0.0061058243059883, 0.0064199987278956, 0.0000000000000000, 0.0494035739313907, 0.0000000000000000, 0.0000000000000000, 0.0977300980425769, 0.0000000000000000, 0.0000000000000000, 0.0092924961699238, 0.0203620584053571, 0.0171970768858957, 0.0977300980425770, 0.0171768454964471, 0.0341431633087197, 0.0254000694760177, 0.0488650490212884, 0.0042942113741118, 0.0085357908271799, 0.0488650490212885, 0.0061058243059883, 0.0256799949115822, 0.0000000000000000, 0.0000000000000000, 0.0254000694760177, 0.0000000000000000, 0.0508001389520353, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0254000694760177, 0.0000000000000000, 0.0004400035036232, 0.0012147906764516, 0.0009809641681319, 0.0138071851046024, 0.0000000000000000, 0.0316811215059028, 0.0000000000000000, 0.0000000000000000, 0.0018356997378421, 0.0073965064431250, 0.0038810845733925, 0.0940073419132750, 0.0313357806377583, 0.0023303788361385, 0.0097185357807154, 0.0061972411364822, 0.0528279010334563, 0.0059270943910571, 0.0206490368123883, 0.0176093003444855, 0.0040190426092311, 0.0107654694290499, 0.0000000000000000, 0.0000000000000000, 0.0274599613301808, 0.0000000000000000, 0.0274599613301808, 0.0128399974557911, 0.0000000000000000, 0.0000000000000000, 0.0254000694760176, 0.0000000000000000, 0.0000000000000000, 0.0254000694760176, 0.0000000000000000, 0.0020077871287887, 0.0000000000000000, 0.0000000000000000, 0.0177961741734280, 0.0000000000000000, 0.0003671292990486, 0.0015647826064445, 0.0059487795180370, 0.0112688687228438, 0.0000000000000000, 0.0000000000000000, 0.0002134954011324, 0.0018199249752804, 0.0022378547334996, 0.0049036693642885, 0.0063588063239205, 0.0139336496231329, 0.0117678693921783, 0.0111460321336458, 0.0001657400828047, 0.0006038012983098, 0.0034313715332893, 0.0020682970591491, 0.0029385062896590, 0.0003549320622510, 0.0002753153925366, 0.0011342962779164, 0.0015646031289944, 0.0064461470507827, 0.0041112440742440, 0.0058409968321762, 0.0002724246508333, 0.0006212378635916, 0.0004790689917299, 0.0013612621447696, 0.0000518687372484, 0.0001780729780017, 0.0010153819077860, 0.0005645512853753, 0.0016041578704297, 0.0001444540565622, 0.0001734580061450, 0.0006066416584268, 0.0007459515778332, 0.0016345564547629, 0.0063588063239205, 0.0139336496231330, 0.0117678693921783, 0.0334380964009373, 0.0017156857666446, 0.0017156857666446, 0.0020682970591491, 0.0117540251586360, 0.0013612621447696, 0.0029385062896590, 0.0015646031289944, 0.0064461470507827, 0.0041112440742440, 0.0233639873287047, 0.0027225242895392, 0.0048124736112892, 0.0058409968321762, 0.0000789529341009, 0.0002913749412336, 0.0024838030698056, 0.0014704281363384, 0.0041781835041622, 0.0003784885679671, 0.0008916977295348, 0.0006708684625998, 0.0038125107574141, 0.0002864273661957, 0.0006124972183726, 0.0019062553787071, 0.0000938912871252, 0.0008279343566019, 0.0004901427121128, 0.0041781835041622, 0.0019062553787071, 0.0002862509059937, 0.0016267490860897, 0.0061843553247977, 0.0087863428219930, 0.0000000000000000, 0.0009459976351016, 0.0011632376687755, 0.0025489290410020, 0.0033053097387001, 0.0072427159200818, 0.0061169426027959, 0.0008918140581994, 0.0005375507653389, 0.0004066406253019, 0.0016753547394496, 0.0010685130498198, 0.0033053097387001, 0.0072427159200818, 0.0061169426027959, 0.0173811341386267, 0.0030548731826928, 0.0060723043696053, 0.0012910816995256, 0.0007643290566682, 0.0021718212385919, 0.0009908722088276, 0.0021718212385919, 0.0002113961851789, 0.0008036575121709, 0.0016267490860897, 0.0061843553247976, 0.0351453712879720, 0.0000000000000000, 0.0000000000000000, 0.0066106194774001, 0.0144854318401636, 0.0122338852055918, 0.0347622682772533, 0.0030548731826928, 0.0060723043696053, 0.0347622682772533, 0.0043436424771838, 0.0091342881093398, 0.0000000000000000, 0.0000000000000000, 0.0087863428219930, 0.0000000000000000, 0.0000000000000000, 0.0173811341386267, 0.0000000000000000, 0.0009459976351016, 0.0011632376687755, 0.0025489290410020, 0.0066106194774001, 0.0144854318401636, 0.0122338852055918, 0.0173811341386267, 0.0017836281163988, 0.0017836281163988, 0.0021502030613555, 0.0061097463653857, 0.0007075845362839, 0.0061097463653857, 0.0016265625012077, 0.0067014189577985, 0.0042740521992793, 0.0121446087392106, 0.0014151690725679, 0.0025015254568760, 0.0121446087392106, 0.0019817444176551, 0.0017181011517284, 0.0037647662541992, 0.0031795888923418, 0.0180694391404688, 0.0015879241428166, 0.0031563859232047, 0.0033053097387001, 0.0072427159200818, 0.0061169426027959, 0.0347622682772533, 0.0061097463653857, 0.0121446087392106, 0.0090347195702344, 0.0008918140581994, 0.0005375507653389, 0.0030548731826928, 0.0014151690725679, 0.0007939620714083, 0.0004066406253019, 0.0016753547394496, 0.0010685130498198, 0.0060723043696053, 0.0007075845362839, 0.0025015254568760, 0.0015781929616023, 0.0033053097387001, 0.0072427159200818, 0.0061169426027959, 0.0347622682772534, 0.0061097463653857, 0.0121446087392106, 0.0090347195702344, 0.0173811341386267, 0.0030548731826928, 0.0060723043696053, 0.0012910816995256, 0.0007643290566682, 0.0043436424771838, 0.0019817444176551, 0.0011289134351567, 0.0021718212385919, 0.0009908722088276, 0.0021718212385919, 0.0008455847407157, 0.0032146300486838, 0.0091342881093398, 0.0000000000000000, 0.0017181011517284, 0.0037647662541992, 0.0031795888923418, 0.0090347195702344, 0.0011289134351567, 0.0182685762186795, 0.0000000000000000, 0.0000000000000000, 0.0180694391404688, 0.0000000000000000, 0.0034362023034568, 0.0075295325083984, 0.0063591777846837, 0.0180694391404688, 0.0031758482856332, 0.0063127718464094, 0.0180694391404688, 0.0015879241428166, 0.0031563859232047, 0.0180694391404688, 0.0022578268703135, 0.0047480061360445, 0.0000000000000000, 0.0000000000000000, 0.0093925007495772, 0.0000000000000000, 0.0015826687120148, 0.0000000000000000, 0.0000000000000000, 0.0015654167915962, 0.0091342881093398, 0.0000000000000000, 0.0000000000000000, 0.0090347195702344, 0.0000000000000000, 0.0180694391404688, 0.0000000000000000, 0.0000000000000000, 0.0017181011517284, 0.0037647662541992, 0.0031795888923418, 0.0090347195702344, 0.0015879241428166, 0.0031563859232047, 0.0180694391404688, 0.0031758482856332, 0.0063127718464094, 0.0093925007495772, 0.0046962503747886, 0.0090347195702344, 0.0011289134351567, 0.0015826687120148, 0.0000000000000000, 0.0000000000000000, 0.0046962503747886, 0.0000000000000000, 0.0000000000000000, 0.0015654167915962, 0.0000550797805507, 0.0000323009538988, 0.0006260310852944, 0.0017283878862832, 0.0013957026654417, 0.0073667541889314, 0.0000000000000000, 0.0010431884042964, 0.0039658530120247, 0.0338066061685317, 0.0000000000000000, 0.0000000000000000, 0.0001210949017489, 0.0002875004438620, 0.0002297934854717, 0.0009258970630336, 0.0002625227415932, 0.0005752494592740, 0.0019588567301616, 0.0078927376450087, 0.0041414663194608, 0.0063588063239205, 0.0139336496231329, 0.0117678693921783, 0.0668761928018746, 0.0008207907446769, 0.0003639478959037, 0.0058770125793180, 0.0001617688236372, 0.0020671329799487, 0.0007234350712577, 0.0116819936643524, 0.0006529522433872, 0.0026309125483362, 0.0013804887731536, 0.0334380964009373, 0.0021196021079735, 0.0046445498743776, 0.0039226231307261, 0.0334380964009373, 0.0058770125793180, 0.0116819936643524, 0.0004077530897651, 0.0001724959265178, 0.0041781835041622, 0.0013927278347207, 0.0003793680346539, 0.0002917175746047, 0.0012165694447161, 0.0024867238213816, 0.0103705518004535, 0.0066130136962936, 0.0187907073156641, 0.0004212130703191, 0.0009568006378614, 0.0007419555882499, 0.0042164952139262, 0.0003201812084379, 0.0021082476069631, 0.0008146655153150, 0.0046353190071754, 0.0025848530905877, 0.0146895863549918, 0.0013020003243007, 0.0030771103363367, 0.0073447931774959, 0.0008676847706203, 0.0008289079404605, 0.0034568506001512, 0.0022043378987645, 0.0187907073156641, 0.0021082476069631, 0.0073447931774959, 0.0002765520987897, 0.0006948458321103, 0.0005031050505679, 0.0042886799521762, 0.0006028522923584, 0.0013448872508074, 0.0014295599840587, 0.0004881167637946, 0.0013476249366078, 0.0010882300963610, 0.0076584827789728, 0.0000000000000000, 0.0175726856439860, 0.0000000000000000, 0.0000000000000000, 0.0010182144096078, 0.0041026477728870, 0.0021527356332143, 0.0173811341386267, 0.0173811341386267, 0.0012925999072107, 0.0053906164326449, 0.0034374468192703, 0.0097674161978932, 0.0010958678393258, 0.0038178260481045, 0.0097674161978932, 0.0022292573306991, 0.0009952205030125, 0.0000000000000000, 0.0076584827789728, 0.0000000000000000, 0.0000000000000000, 0.0195348323957864, 0.0175726856439860, 0.0000000000000000, 0.0000000000000000, 0.0347622682772533, 0.0000000000000000, 0.0000000000000000, 0.0010182144096078, 0.0041026477728870, 0.0021527356332143, 0.0347622682772534, 0.0021917356786517, 0.0076356520962090, 0.0173811341386266, 0.0173811341386266, 0.0012925999072107, 0.0053906164326449, 0.0034374468192702, 0.0195348323957864, 0.0021917356786517, 0.0076356520962090, 0.0050771063366699, 0.0097674161978932, 0.0010958678393258, 0.0038178260481045, 0.0097674161978932, 0.0022292573306991, 0.0039808820120501, 0.0000000000000000, 0.0000000000000000, 0.0050771063366699, 0.0000000000000000, 0.0101542126733398, 0.0000000000000000, 0.0050771063366699, 0.0004227923703579, 0.0016073150243419, 0.0091342881093398, 0.0000000000000000, 0.0000000000000000, 0.0017181011517284, 0.0037647662541992, 0.0031795888923418, 0.0090347195702344, 0.0007939620714083, 0.0015781929616023, 0.0090347195702344, 0.0011289134351567, 0.0023740030680223, 0.0000000000000000, 0.0000000000000000, 0.0046962503747886, 0.0091342881093397, 0.0000000000000000, 0.0000000000000000, 0.0180694391404688, 0.0000000000000000, 0.0000000000000000, 0.0017181011517284, 0.0037647662541992, 0.0031795888923418, 0.0180694391404688, 0.0031758482856332, 0.0063127718464093, 0.0046962503747886, 0.0090347195702343, 0.0007939620714083, 0.0015781929616023, 0.0090347195702344, 0.0011289134351567, 0.0094960122720890, 0.0000000000000000, 0.0000000000000000, 0.0093925007495772, 0.0000000000000000, 0.0093925007495772, 0.0000000000000000, 0.0000000000000000, 0.0093925007495772, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0093925007495772, 0.0000000000000000, 0.0000150471326846, 0.0000775226022724, 0.0002513354405944, 0.0001520678888305, 0.0001227974110749, 0.0028566526060033, 0.0000000000000000, 0.0049111694592876, 0.0000000000000000, 0.0000000000000000, 0.0112688687228438, 0.0000000000000000, 0.0000000000000000, 0.0003290314785747, 0.0017113912329929, 0.0004858354191294, 0.0334380964009372, 0.0111460321336457, 0.0001696920500192, 0.0022141991703643, 0.0007757726450051, 0.0187907073156641, 0.0002410286481384, 0.0021258378166811, 0.0062635691052213, 0.0002812104195815, 0.0006919244402265, 0.0040116060532550, 0.0022277308249664, 0.0189901184992955, 0.0015753897501938, 0.0074386601575528, 0.0063300394997651, 0.0014734674792886, 0.0022273335271715, 0.0000000000000000, 0.0000000000000000, 0.0098710702004980, 0.0000000000000000, 0.0098710702004980, 0.0019904410060251, 0.0000000000000000, 0.0000000000000000, 0.0050771063366699, 0.0000000000000000, 0.0000000000000000, 0.0050771063366699, 0.0000000000000000, 0.0023740030680222, 0.0000000000000000, 0.0000000000000000, 0.0046962503747886, 0.0000000000000000, 0.0000000000000000, 0.0046962503747885, 0.0000000000000000, 0.0004714022849150, 0.0000000000000000, 0.0000000000000000, 0.0089050663613568, 0.0000000000000000};

    yscaler.mean = {0.0087779223946766};
    yscaler.std  = {0.0266654918551303};
}

NNPIP::~NNPIP()
{
    delete yij;
    delete mono;
    delete poly;
}

double NNPIP::pes(std::vector<double> const& x) 
{
    double drx, dry, drz;

    size_t k = 0;

    for (size_t i = 0; i < NATOMS; ++i) {
        for (size_t j = i + 1; j < NATOMS; ++j) {

            drx = ATOMX(x, i) - ATOMX(x, j);
            dry = ATOMY(x, i) - ATOMY(x, j);
            drz = ATOMZ(x, i) - ATOMZ(x, j);

            yij[k] = std::sqrt(drx*drx + dry*dry + drz*drz);
            yij[k] = std::exp(-yij[k]/a0);
            k++;
        }
    }

    assert((k == NDIS) && ": ERROR: the morse variables vector is not filled properly.");

    c_evmono(yij, mono);
    c_evpoly(mono, poly);

    xscaler.transform(poly, NPOLY);

    t = torch::from_blob(poly, {static_cast<long int>(NPOLY)}, torch::kDouble);
    double ytr = model.forward({t}).toTensor().item<double>();

    return ytr * yscaler.std[0] + yscaler.mean[0];
}

double internal_pes(NNPIP & pes, double R, double PH1, double TH1, double PH2, double TH2)
// internal coordinates -> cartesian coordinates
// call NNPIP.pes(cartesian coordinates)
{
    static std::vector<double> cart(21);

    cart[0] =  1.193587416; cart[1]  =  1.193587416; cart[2]  = -1.193587416; // H1 
    cart[3] = -1.193587416; cart[4]  = -1.193587416; cart[5]  = -1.193587416; // H2
    cart[6] = -1.193587416; cart[7]  =  1.193587416; cart[8]  =  1.193587416; // H3 
    cart[9] =  1.193587416; cart[10] = -1.193587416; cart[11] =  1.193587416; // H4 

    // N1
    cart[12] = R * std::sin(TH1) * std::cos(PH1) - NN_BOND_LENGTH * std::cos(PH2) * std::sin(TH2);
    cart[13] = R * std::sin(TH1) * std::sin(PH1) - NN_BOND_LENGTH * std::sin(PH2) * std::sin(TH2);
    cart[14] = R * std::cos(TH1)                 - NN_BOND_LENGTH * std::cos(TH2);
    
    // N2
    cart[15] = R * std::sin(TH1) * std::cos(PH1) + NN_BOND_LENGTH * std::cos(PH2) * std::sin(TH2);
    cart[16] = R * std::sin(TH1) * std::sin(PH1) + NN_BOND_LENGTH * std::sin(PH2) * std::sin(TH2);
    cart[17] = R * std::cos(TH1)                 + NN_BOND_LENGTH * std::cos(TH2);
    
    cart[18] = 0.0; cart[19] = 0.0; cart[20] = 0.0;                           // C  

    return pes.pes(cart);
}

int main(int argc, char* argv[])
{
    std::cout << std::fixed << std::setprecision(16);
    
    const size_t NATOMS = 7;
    NNPIP nn_pes(NATOMS, "model.pt");

    AI_PES_ch4_n2 symm_pes;
    symm_pes.init();

    const double deg = M_PI / 180.0;
    double PH1 = 47.912 * deg;
    double TH1 = 56.167 * deg;
    double PH2 = 0.0    * deg;
    double TH2 = 135.0  * deg;

    std::vector<double> Rv = linspace(4.5, 30.0, 500);

    for (double R : Rv) {
        double nnval   = internal_pes(nn_pes, R, PH1, TH1, PH2, TH2) * HTOCM;
        double symmval = symm_pes.pes(R, PH1, TH1, PH2, TH2);

        std::cout << R << " " << nnval << " " << symmval << "\n"; 
    }

    return 0;
}

