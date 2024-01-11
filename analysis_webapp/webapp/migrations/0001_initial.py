# Generated by Django 4.2.6 on 2024-01-11 18:49

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Multimodal",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("created", models.DateTimeField(auto_now=True)),
                (
                    "model_name",
                    models.CharField(blank=True, default="", max_length=100),
                ),
                (
                    "model_path",
                    models.CharField(blank=True, default="", max_length=100),
                ),
            ],
        ),
        migrations.CreateModel(
            name="Predictions",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("created", models.DateField(auto_now=True)),
                (
                    "multimodal",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="webapp.multimodal",
                    ),
                ),
            ],
        ),
    ]
