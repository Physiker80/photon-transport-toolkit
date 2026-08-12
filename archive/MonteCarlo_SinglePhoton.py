# Archived, not maintained: the literal original single-photon script this
# whole project grew from. Kept for historical record only. The current,
# validated, tested engine is src/photon_transport_toolkit/monte_carlo.py --
# see PROJECT_REPORT.md Section 1 for how this script relates to it.
import numpy as np
import matplotlib.pyplot as plt

# المعاملات البصرية من الدليل
mu_a = 0.5   # معامل الامتصاص (1/cm)
mu_s = 15.0  # معامل الاستطارة (1/cm)
g = 0.90     # معامل التباين
mu_t = mu_a + mu_s

# تهيئة الفوتون عند نقطة السقوط
x, y, z = 0.0, 0.0, 0.0
ux, uy, uz = 0.0, 0.0, 1.0  # الاتجاه عمودي نحو الأسفل
weight = 1.0

positions_x = [x]
positions_z = [z]

# تتبع المسار حتى يتلاشى وزن الفوتون أو يخرج من السطح
while weight > 0.0001:
    # 1. تحديد حجم الخطوة
    xi = np.random.random()
    s = -np.log(xi) / mu_t
    
    # 2. تحريك الفوتون
    x += ux * s
    y += uy * s
    z += uz * s
    
    # التحقق من خروج الفوتون من السطح (الانعكاس الانتشاري)
    if z <= 0:
        positions_x.append(x)
        positions_z.append(z)
        break
        
    positions_x.append(x)
    positions_z.append(z)
    
    # 3. الامتصاص (تقليل الوزن)
    weight -= weight * (mu_a / mu_t)
    
    # 4. الاستطارة وتغيير الاتجاه
    xi1 = np.random.random()
    if g == 0:
        cost = 2 * xi1 - 1
    else:
        temp = (1 - g**2) / (1 - g + 2 * g * xi1)
        cost = (1 + g**2 - temp**2) / (2 * g)
    sint = np.sqrt(max(0, 1 - cost**2))
    
    xi2 = np.random.random()
    psi = 2 * np.pi * xi2
    cosp = np.cos(psi)
    sinp = np.sin(psi)
    
    # 5. تحديث جيوب التمام الاتجاهية
    if abs(uz) > 0.99999: # السقوط العمودي
        ux_new = sint * cosp
        uy_new = sint * sinp
        uz_new = np.sign(uz) * cost
    else:
        temp = np.sqrt(1 - uz**2)
        ux_new = sint * (ux * uz * cosp - uy * sinp) / temp + ux * cost
        uy_new = sint * (uy * uz * cosp + ux * sinp) / temp + uy * cost
        uz_new = -sint * cosp * temp + uz * cost
        
    ux, uy, uz = ux_new, uy_new, uz_new

# تحويل الوحدات من سنتيمتر إلى ميكرومتر للرسم البياني
pos_x_um = np.array(positions_x) * 10000
pos_z_um = np.array(positions_z) * 10000

# إعداد الرسم البياني
plt.figure(figsize=(8, 6))
plt.plot(pos_x_um, pos_z_um, color='red', marker='.', linestyle='-', markersize=5)

# تحديد نقطة الدخول بوضوح
plt.annotate('', xy=(0, 0), xytext=(0, -500),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))

# إذا خرج الفوتون، أضف سهم خروج
if pos_z_um[-1] <= 0:
    dx = pos_x_um[-1] - pos_x_um[-2]
    dz = pos_z_um[-1] - pos_z_um[-2]
    plt.annotate('', xy=(pos_x_um[-1] + dx*2, pos_z_um[-1] + dz*2), xytext=(pos_x_um[-1], pos_z_um[-1]),
                arrowprops=dict(facecolor='black', shrink=0, width=1.5, headwidth=8))

plt.gca().invert_yaxis() # قلب المحور الصادي ليتوافق مع العمق
plt.xlabel('position x ($\mu$m)')
plt.ylabel('depth z ($\mu$m)')
plt.title('Monte Carlo Simulation of a Single Photon')
plt.grid(True, linestyle='--', color='blue', alpha=0.3)

# تنسيق المحاور ليشابه الصورة المرفقة
plt.gca().spines['top'].set_color('blue')
plt.gca().spines['bottom'].set_color('blue')
plt.gca().spines['left'].set_color('blue')
plt.gca().spines['right'].set_color('blue')
plt.tick_params(axis='both', colors='blue', direction='in', length=6)

plt.show()