from django.contrib.admin.views.decorators import staff_member_required

from django.shortcuts import render, redirect, get_object_or_404
from .models import UploadedFile, PatientData as PatientDataModel
from django.http import HttpResponse
from django.urls import reverse 
from django.contrib import messages


def print_item_view(request, item_id):
    item = get_object_or_404(UploadedFile, id=item_id)

    patient_to_print = UploadedFile.objects.filter(id=item_id)

    context = {'patient_to_print': patient_to_print}

    return render(request, "inference_print.html", context)

def delete_item_view(request, item_id):
    item = get_object_or_404(UploadedFile, id=item_id)

    patient_to_delete = UploadedFile.objects.filter(id=item_id)
                
    context = {'patient_to_delete': patient_to_delete}

    action = request.POST.get('action')

    if action == 'verify':
    	selected_patient = request.POST.get('idPrint')
    	patient_to_delete = UploadedFile.objects.filter(id=selected_patient)

    	entered_password = request.POST.get('password')
    	user = request.user
    	if user.check_password(entered_password):
    		patient = get_object_or_404(UploadedFile, id=selected_patient)
    		patient.delete()
    		messages.success(request, 'Data Berhasil dihapus')
    		return redirect('../admin/hibahs/uploadedfile/')
    	else:
    		messages.error(request, 'Password salah')
    		return render(request, "delete_image.html", {'password_error': True, 'patient_to_delete': patient_to_delete})
    
    return render(request, "delete_image.html",  context)










