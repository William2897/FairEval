from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from api.models import UserRole, Professor
from data_processing.dept_mapping import disciplines_dict

class Command(BaseCommand):
    help = 'Create a user with specified role (admin or academic) or create a user account for existing professor'

    def add_arguments(self, parser):
        parser.add_argument('username', type=str, nargs='?', default=None)
        parser.add_argument('password', type=str)
        parser.add_argument('role', type=str, choices=['ADMIN', 'ACADEMIC'], nargs='?', default=None)
        valid_disciplines = list(disciplines_dict.keys())
        parser.add_argument('--discipline', type=str, 
                          choices=valid_disciplines,
                          help=f'Discipline {valid_disciplines}', 
                          default=None)
        parser.add_argument('--email', type=str, default='')
        parser.add_argument('--first_name', type=str, default='')
        parser.add_argument('--last_name', type=str, default='')
        parser.add_argument('--professor_id', type=str, default=None)

    def handle(self, *args, **options):
        username = options['username']
        password = options['password']
        role = options['role']
        discipline = options['discipline']
        professor_id = options['professor_id']

        if professor_id:
            try:
                # Get existing professor data
                professor = Professor.objects.get(professor_id=professor_id)
                
                # Create user with professor's existing data
                user = User.objects.create_user(
                    username=professor_id,
                    password=password,
                    email=f"{professor.first_name.lower()}.{professor.last_name.lower()}@institution.edu",
                    first_name=professor.first_name,
                    last_name=professor.last_name
                )

                # Get discipline from professor's department
                discipline = professor.department.discipline

                # Create UserRole
                UserRole.objects.create(
                    user=user,
                    role='ACADEMIC',
                    discipline=discipline
                )

                self.stdout.write(
                    self.style.SUCCESS(
                        f'Successfully created user for professor "{professor.first_name} {professor.last_name}" '
                        f'with discipline "{discipline}"'
                    )
                )

            except Professor.DoesNotExist:
                self.stderr.write(f'Professor with ID {professor_id} not found in database')
            except Exception as e:
                self.stderr.write(f'Error creating user: {str(e)}')
            return

        if role == 'ACADEMIC':
            if not discipline:
                self.stderr.write('Discipline is required for academic role')
                return
            if discipline not in disciplines_dict:
                self.stderr.write(f'Invalid discipline. Choose from: {list(disciplines_dict.keys())}')
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
                UserRole.objects.create(
                    user=user, 
                    role='ACADEMIC',
                    discipline=options['discipline']
                )
                self.stdout.write(self.style.SUCCESS(f'Successfully created academic user "{username}" in discipline "{options["discipline"]}"'))

        except Exception as e:
            self.stderr.write(f'Error creating user: {str(e)}')