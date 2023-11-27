from django.db import models

# Create your models here.
class PatientData(models.Model):
	GENDER_CHOICES = ( 
		('L', 'Laki-laki'), 
		('P', 'Perempuan'), 
	)

	patientName = models.CharField('Nama Pasien', max_length = 225)
	patientBirthDate = models.DateField('Tanggal Lahir', auto_now_add=False, auto_now=False)
	patientGender = models.CharField('Jenis Kelamin', max_length = 1, choices=GENDER_CHOICES, default='L')
	patientID = models.CharField('No Pasien', max_length = 16, unique=True, default=None)

	def __str__(self):
		return self.patientName

	class Meta:
		verbose_name_plural = 'Data Pasien'  # Change the display name here
	
	

class UploadedFile(models.Model):
	patientName = models.ForeignKey(PatientData, on_delete=models.CASCADE)
	file = models.FileField(upload_to='uploads/')
	diagnosisResult = models.TextField(null=True, blank=True)
	limeImageResult = models.FileField(default=None)
	imageDate = models.DateField('Tanggal Image diambil (mm/dd/yyyy)', auto_now_add=False, auto_now=False)
	uploaded_at = models.DateTimeField(auto_now_add=True)

	def get_float_list(self):
		float_values = self.diagnosisResult[1:-1].split()

		return [float(value) for value in float_values]

	def __str__(self):
		return str(self.get_float_list())


	def patient_data(self):
		return self.patientName

	

	



class UploadedImage(models.Model):
    image = models.ImageField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)






