from django.contrib import admin, messages
from django.utils.html import format_html
from django.urls import reverse, path
from django.views.generic import TemplateView
from django.contrib.auth.mixins import PermissionRequiredMixin
from django.views.generic.detail import SingleObjectMixin, DetailView
from django import forms
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse

from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _
from django.core.validators import RegexValidator
from datetime import datetime




from hibahs.LimeViz import get_image, get_pil_transform, batch_predict, batch_explaination
from skimage.segmentation import mark_boundaries
import cv2
import os







admin.site.site_header = 'InDRI (Intelligent Diagnosis of Radiology Images)'
# Register your models here.
from .models import UploadedFile, PatientData as PatientDataModel

class PatientUpload(admin.StackedInline):
	model = UploadedFile


class PatientDataForm(forms.ModelForm):
    current_date = datetime.now().date()


    class Meta:
        model = PatientDataModel
        fields = '__all__'

   

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['patientID'].disabled = True

        # If the instance is not provided (i.e., when creating a new patient), generate no_pasien
        if not self.instance.pk:
            latest_patient = PatientDataModel.objects.order_by('-patientID').first()
            if latest_patient:
                numeric_part = int(latest_patient.patientID[3:])
                next_numeric_part = numeric_part + 1
                self.initial['patientID'] = f'PAS{next_numeric_part:03}'
            else:
                self.initial['patientID'] = 'PAS001'

    patientName = forms.CharField(
        validators=[RegexValidator(regex=r'^[a-zA-Z\s]*$', message='Enter only alphabetic characters.')],
        label='Nama Pasien'
    )

    patientBirthDate = forms.DateField(
        widget=forms.DateInput(attrs={'type': 'date', 'max': str(current_date)}),
        label='Tanggal Lahir Pasien',
    )


class PatientData(admin.ModelAdmin):
   form = PatientDataForm
   list_display = ('patientID', 'patientName', 'patientBirthDate', 'patientGender') 


   
admin.site.register(PatientDataModel, PatientData)





class PatientDetailView(PermissionRequiredMixin, DetailView):
    permission_required = "hibahs.view_uploadedimage"
    template_name = "patient_with_image.html"
    model = PatientDataModel

    def get_context_data(self, **kwargs):
        return {
            **super().get_context_data(**kwargs),
            **admin.site.each_context(self.request),
            "opts": self.model._meta,
        }
    
    def post(self, request, *args, **kwargs):
       

        if request.method == 'POST':
            action = request.POST.get('action')

            if action == 'inference':
                selected_patient = request.POST.getlist('selected_files')
                if not selected_patient:

                    return HttpResponse(f'<script>alert("Silahkan Pilih Salah satu image yang mau didiagnosa"); window.history.back();</script>')

                else:
                    filenames = []
                    ids = []

                    for item in selected_patient:
                        parts = item.split(', ')  # Split each item by ', '
                        if len(parts) == 2:
                            filenames.append(parts[0])
                            ids.append(parts[1])

                    base_directory = 'media/'

                    image = [get_image(base_directory + path) for path in filenames]


                    pill_transform = get_pil_transform()
                    image_transf = [pill_transform(img) for img in image]

                    probs = batch_predict(image_transf)
                    # print('prob',probs)
                    temps, masks = batch_explaination(image_transf)
                    limeImage=[]
                    for i in range(len(temps)):
                        marked_img = mark_boundaries(temps[i], masks[i])
                        conv_img = (marked_img * 255).astype('uint8')
                        bgr_img =  cv2.cvtColor(conv_img, cv2.COLOR_RGB2BGR)
                        limeImage = "cam_" + str(ids[i]) + ".jpg"
                        cv2.imwrite(os.path.join('media/static', limeImage), bgr_img)

                    threshold = 0.14

                    # probs = [[6.95283189e-02, 1.94228895e-03, 1.14850536e-01, 8.13678920e-01], [9.99474347e-01, 5.25628682e-04, 1.26138291e-08, 3.79446767e-08]]

                    combined_data = []

                    for i in range(len(probs)):
                        UploadedFile.objects.filter(id=ids[i]).update(diagnosisResult=probs[i], limeImageResult="cam_" + str(ids[i]) + ".jpg")
                        combined_data.append({
                            'result': probs[i],
                            'filename': filenames[i],
                            'threshold': threshold,
                            'limeImage': ids[i],
                        })
                
                    # return HttpResponse(f'ID: {filenames}')
                    return render(request, "inference_output.html",   {'combined_data': combined_data})

            

        return render(request, self.template_name)



class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True

class MultipleFileField(forms.FileField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput())
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        single_file_clean = super().clean
        if isinstance(data, list):
            # Allow processing multiple files
            result = [single_file_clean(d, initial) for d in data]
        else:
            result = single_file_clean(data, initial)
        return result


class PatientChoiceField(forms.ModelChoiceField):
    def label_from_instance(self, obj):
        return f'{obj.patientID} - {obj.patientName}'


def validate_file_type(value):
    if not value.name.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise ValidationError(_('Silahkan upload gambar dengan tipe file PNG, JPEG atau JPG'))



class UploadFileForm(forms.Form):
    patientName = PatientChoiceField(
        queryset= PatientDataModel.objects.all(),
        empty_label='Pilih Nama Pasien',  # Remove the empty label (optional)
        to_field_name='id',
        label="Nama Pasien"
    )
    
    files = MultipleFileField(validators=[validate_file_type])

    current_date = datetime.now().date()

    imageDate = forms.DateField(
        widget=forms.DateInput(attrs={'type': 'date', 'max': str(current_date)}),
        label='Tanggal Diambil',
    )


    class Meta:
        model = UploadedFile
        fields = ['file', 'patientName', 'imageDate']

    

        


 

class UploadedFile_list(admin.ModelAdmin):
   list_display = ('get_patient_name', 'detail')
   list_display_links = None



   def has_add_permission(self, request):
        return False

   def has_change_permission(self, request):
        return False 
   
   def get_urls(self):
        urls = super().get_urls()

        custom_urls = [
            # Define your custom URL patterns here
            path('upload-image/', self.upload_image),
            path('<pk>/detail', self.admin_site.admin_view(PatientDetailView.as_view()), name='patient_detail'),
        ]

        return custom_urls + urls

        
   

   def upload_image(self, request):

        files = UploadedFile.objects.all()

        # Create a dictionary to store unique entries based on a specific field (e.g., 'field_to_check')
        unique_entries = {}

        # Iterate through the queryset and store unique entries in the dictionary
        for file in files:
            key = file.patientName_id  # Use the field you want to check for uniqueness
            if key not in unique_entries:
                unique_entries[key] = file

        if request.method == 'POST':
            form = UploadFileForm(request.POST, request.FILES)
            if form.is_valid():
                selected_patient = form.cleaned_data['patientName']
                selected_patient_id = selected_patient.id
                selected_tanggaldiambil = form.cleaned_data['imageDate']
                for uploaded_file in request.FILES.getlist('files'):
                    UploadedFile.objects.create(file=uploaded_file, patientName_id=selected_patient_id, imageDate=selected_tanggaldiambil)
                return redirect('../')
        else:
            form = UploadFileForm()

        return render(request, "upload_and_display.html", {'form': form, 'files': unique_entries.values()})
        
   

   def get_queryset(self, request):
        queryset = super().get_queryset(request)
        # Create a dictionary to store the latest record for each unique patient
        latest_records = {}
        
        for file in queryset:
            patient_id = file.patientName.id
            if patient_id not in latest_records:
                latest_records[patient_id] = file
            else:
                # Check if this record has a more recent uploaded_at date
                if file.uploaded_at > latest_records[patient_id].uploaded_at:
                    latest_records[patient_id] = file
        
        # Convert the dictionary values back to a queryset
        return UploadedFile.objects.filter(pk__in=[record.id for record in latest_records.values()])

   def get_patient_name(self, obj):
   		return obj.patientName.patientName  # Replace 'name' with the actual field name in your Patient model
   get_patient_name.short_description = 'Patient Name'  # This sets the column header text in the admin list view



   def detail(self, obj: PatientDataModel) -> str:
   		foreign_key_value = obj.patientName_id
   		url = reverse("admin:patient_detail", args=[foreign_key_value])
   		return format_html(f'<a href="{url}">📝</a>')

  

   
   

        
        



   


	

admin.site.register(UploadedFile, UploadedFile_list)








