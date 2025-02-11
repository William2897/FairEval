from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from api.models import UserRole, Department

class Command(BaseCommand):
    help = 'Create a user with specified role (admin or academic)'

    def add_arguments(self, parser):
        parser.add_argument('username', type=str)
        parser.add_argument('password', type=str)
        parser.add_argument('role', type=str, choices=['ADMIN', 'ACADEMIC'])
        parser.add_argument('--department', type=str, help='Department name (required for academic role)', default=None)
        parser.add_argument('--email', type=str, default='')
        parser.add_argument('--first_name', type=str, default='')
        parser.add_argument('--last_name', type=str, default='')

    def handle(self, *args, **options):
        username = options['username']
        password = options['password']
        role = options['role']
        
        if role == 'ACADEMIC' and not options['department']:
            self.stderr.write('Department is required for academic role')
            return

        try:
            user = User.objects.create_user(
                username=username,
                email=options['email'],
                password=password,
                first_name=options['first_name'],
                last_name=options['last_name']
            )

            if role == 'ADMIN':
                user.is_staff = True
                user.is_superuser = True
                user.save()
                UserRole.objects.create(user=user, role='ADMIN')
                self.stdout.write(self.style.SUCCESS(f'Successfully created admin user "{username}"'))
            else:
                department, _ = Department.objects.get_or_create(name=options['department'])
                UserRole.objects.create(user=user, role='ACADEMIC', department=department)
                self.stdout.write(self.style.SUCCESS(f'Successfully created academic user "{username}" in department "{department.name}"'))

        except Exception as e:
            self.stderr.write(f'Error creating user: {str(e)}')